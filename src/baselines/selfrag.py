#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-RAG baseline.

This script runs a Self-RAG-style pipeline:
1. Judge retrieved documents.
2. Select useful documents.
3. Generate an answer.
4. Optionally verify and revise the answer.
5. Evaluate predictions.

Supported dataset types:
- general QA
- FEVER
- TruthfulQA multiple-choice evaluation

TruthfulQA requires an auxiliary dev.jsonl file containing choices and
gold answer indices.
"""

import argparse
import json
import os
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig

try:
    from transformers import Mxfp4Config
except ImportError:
    Mxfp4Config = None


def strip_outer_quotes(text: str) -> str:
    text = str(text).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].strip()
    return text


def ensure_list_of_str(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, dict):
        for key in ["answers", "gold_answers", "golden_answers", "ground_truth", "text", "answer"]:
            if key in value:
                return ensure_list_of_str(value[key])
        return [json.dumps(value, ensure_ascii=False)]

    raw_list = value if isinstance(value, list) else [value]
    output = []

    for item in raw_list:
        text = str(item).strip()
        if not text:
            continue

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    output.extend(ensure_list_of_str(parsed))
                    continue
            except Exception:
                pass

        if "|" in text:
            output.extend([part.strip() for part in text.split("|") if part.strip()])
        else:
            output.append(text)

    return list(dict.fromkeys(output))


def normalize_answer(text: Any) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punctuation(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    def fix_whitespace(s: str) -> str:
        return " ".join(s.split())

    if text is None:
        text = ""

    text = str(text).lower()
    return fix_whitespace(remove_articles(remove_punctuation(text)))


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_sub_em(gold: str, pred: str) -> int:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    return int(bool(gold_norm) and gold_norm in pred_norm)


def compute_f1(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()

    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return float(gold_tokens == pred_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_qa_metrics(predictions: List[str], golds: List[List[str]]) -> Dict[str, float]:
    em_total = 0.0
    f1_total = 0.0
    acc_total = 0.0
    valid = 0

    for pred, gold_list in zip(predictions, golds):
        gold_list = ensure_list_of_str(gold_list)
        if not gold_list:
            continue

        valid += 1
        em_total += max(compute_exact(gold, pred) for gold in gold_list)
        f1_total += max(compute_f1(gold, pred) for gold in gold_list)
        acc_total += max(compute_sub_em(gold, pred) for gold in gold_list)

    if valid == 0:
        return {"em": 0.0, "f1": 0.0, "acc": 0.0}

    return {
        "em": em_total / valid,
        "f1": f1_total / valid,
        "acc": acc_total / valid,
    }


def normalize_fever_label(text: str) -> str:
    text = str(text).strip().upper()
    if "SUPPORTS" in text:
        return "SUPPORTS"
    if "REFUTES" in text:
        return "REFUTES"
    return text


def calculate_fever_metrics(predictions: List[str], golds: List[List[str]]) -> Dict[str, float]:
    correct = 0
    valid = 0

    for pred, gold_list in zip(predictions, golds):
        gold_list = ensure_list_of_str(gold_list)
        if not gold_list:
            continue

        pred_label = normalize_fever_label(pred)
        gold_labels = {normalize_fever_label(gold) for gold in gold_list}

        valid += 1
        correct += int(pred_label in gold_labels)

    acc = correct / valid if valid > 0 else 0.0
    return {"em": acc, "f1": acc, "acc": acc}


def extract_doc_text(doc: Any) -> str:
    if doc is None:
        return ""

    if isinstance(doc, str):
        return doc

    if isinstance(doc, dict):
        for key in [
            "contents",
            "content",
            "text",
            "passage",
            "ctx",
            "document",
            "body",
            "context",
            "chunk",
            "paragraph",
            "sentence",
            "fact",
        ]:
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(doc, ensure_ascii=False)

    return str(doc)


def extract_contexts(item: Dict[str, Any]) -> List[Any]:
    for key in [
        "selected_contexts",
        "ctxs",
        "candidates",
        "retrieved_results",
        "retrieved_contexts",
        "contexts",
        "docs",
        "documents",
        "passages",
    ]:
        value = item.get(key)
        if isinstance(value, list):
            return value
    return []


def truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[TRUNCATED]"


def calculate_recall(data: List[Dict[str, Any]], top_k: int = 5) -> float:
    hits = []
    for item in data:
        answers = ensure_list_of_str(item.get("ground_truth", item.get("answers", [])))
        contexts = extract_contexts(item)[:top_k]
        context_text = " ".join(extract_doc_text(doc) for doc in contexts)
        context_norm = normalize_answer(context_text)
        hit = any(normalize_answer(ans) in context_norm for ans in answers if normalize_answer(ans))
        hits.append(1 if hit else 0)

    return float(np.mean(hits)) if hits else 0.0


def load_data(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def idx_to_letter(index: int) -> str:
    return chr(ord("A") + index)


def letter_to_idx(text: str) -> Optional[int]:
    if not text:
        return None

    parts = re.split(r"assistant", str(text), flags=re.IGNORECASE)
    target_text = parts[-1] if len(parts) > 1 else str(text)

    found = re.findall(r"\b([A-Z])\b", target_text)
    if found:
        return ord(found[-1].upper()) - ord("A")

    return None


def pick_choice_by_text(prediction: str, choices: List[str]) -> Optional[int]:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return None

    hits = []
    for index, choice in enumerate(choices):
        choice_norm = normalize_answer(choice)
        if choice_norm and choice_norm in pred_norm:
            hits.append((len(choice_norm), index))

    if hits:
        hits.sort(reverse=True)
        return hits[0][1]

    pred_tokens = set(pred_norm.split())
    if not pred_tokens:
        return None

    best_index = None
    best_score = -1.0

    for index, choice in enumerate(choices):
        choice_tokens = set(normalize_answer(choice).split())
        if not choice_tokens:
            continue

        score = len(pred_tokens & choice_tokens) / max(1, len(choice_tokens))
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def build_truthfulqa_choice_map(dev_jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    rows = load_data(dev_jsonl_path)
    choice_map = {}

    for row in rows:
        key = str(row.get("id", row.get("query_id", ""))).strip()
        if not key:
            continue

        choices = row.get("choices", None)
        gold_idx = row.get("golden_answers", None)

        if not isinstance(choices, list) or len(choices) == 0:
            continue

        if not isinstance(gold_idx, list):
            gold_idx = []

        choices = [str(choice).strip() for choice in choices if str(choice).strip()]

        gold_idx_clean = []
        gold_texts = []

        for idx in gold_idx:
            try:
                idx = int(idx)
                if 0 <= idx < len(choices):
                    gold_idx_clean.append(idx)
                    gold_texts.append(choices[idx])
            except Exception:
                pass

        choice_map[key] = {
            "choices": choices,
            "gold_idx": gold_idx_clean,
            "gold_texts": gold_texts,
        }

    return choice_map


def compute_truthfulqa_acc(
    predictions: List[str],
    meta: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    correct = 0
    details = []

    for pred, item in zip(predictions, meta):
        choices = item["choices"]
        gold_idx = item["gold_idx"]

        pred_idx = letter_to_idx(pred)

        if pred_idx is not None and (pred_idx < 0 or pred_idx >= len(choices)):
            pred_idx = None

        if pred_idx is None:
            pred_idx = pick_choice_by_text(pred, choices)

        if pred_idx is not None and gold_idx:
            is_correct = int(int(pred_idx) in set(gold_idx))
        else:
            gold_texts = ensure_list_of_str(item.get("gold_texts", item.get("ground_truth", [])))
            pred_norm = normalize_answer(pred)
            is_correct = int(any(normalize_answer(gold) in pred_norm for gold in gold_texts))

        correct += is_correct

        details.append(
            {
                "query_id": item["query_id"],
                "question": item["question"],
                "choices": choices,
                "gold_idx": gold_idx,
                "gold_texts": item.get("gold_texts", []),
                "raw_prediction": pred,
                "pred_choice_idx": pred_idx,
                "pred_choice_letter": idx_to_letter(pred_idx) if pred_idx is not None else None,
                "pred_choice_text": choices[pred_idx] if pred_idx is not None and 0 <= pred_idx < len(choices) else None,
                "is_correct": is_correct,
            }
        )

    acc = correct / len(meta) if meta else 0.0
    return acc, details


class LocalGenerator:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        load_in_4bit: bool = False,
        quantization_mode: str = "none",
    ):
        if tokenizer_path is None:
            tokenizer_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = (getattr(config, "model_type", "") or "").lower()
        except Exception:
            model_type = ""

        model_path_lower = model_path.lower()

        if "qwen" in model_type or "qwen" in model_path_lower:
            self.model_family = "qwen"
        elif "llama" in model_type or "llama" in model_path_lower:
            self.model_family = "llama"
        elif "gpt" in model_type or "gpt" in model_path_lower or "oss" in model_path_lower:
            self.model_family = "gpt"
        else:
            self.model_family = "auto"

        if load_in_4bit:
            quantization_mode = "bnb4"

        if quantization_mode == "mxfp4":
            if Mxfp4Config is None:
                raise ImportError(
                    "Mxfp4Config is not available. Please upgrade transformers or use "
                    "--quantization_mode none/bnb4."
                )
            quant_config = Mxfp4Config(dequantize=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
            ).eval()

        elif quantization_mode == "bnb4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
            ).eval()

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                trust_remote_code=True,
            ).eval()

        self.harmony_final_prefix = "<|channel|>final<|message|>"
        self.debug_raw = False
        self.debug_counter = 0
        self.debug_limit = 0

    def build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = []

        if self.model_family == "gpt" and system_prompt:
            messages.append({"role": "developer", "content": system_prompt})
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        if self.model_family == "qwen":
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                pass

        if self.model_family == "gpt":
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort="medium",
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return rendered + self.harmony_final_prefix

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _safe_token_id(self, token: str):
        try:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
        except Exception:
            return None

        if token_id is None:
            return None
        if self.tokenizer.unk_token_id is not None and token_id == self.tokenizer.unk_token_id:
            return None
        return token_id

    def get_stop_ids(self):
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0] if eos_id else None

        stop_ids = []
        for token_id in [
            eos_id,
            self._safe_token_id("<|eot_id|>"),
            self._safe_token_id("<|im_end|>"),
        ]:
            if token_id is not None and token_id not in stop_ids:
                stop_ids.append(token_id)

        return stop_ids

    def get_bad_words_ids(self):
        bad_words = []

        for token in [
            "<|im_start|>",
            "<|start_header_id|>",
            "<|channel|>",
            "<|message|>",
        ]:
            token_id = self._safe_token_id(token)
            if token_id is not None:
                bad_words.append([token_id])

        return bad_words if bad_words else None

    @staticmethod
    def response_generation(full_answer: str) -> str:
        text = str(full_answer).replace("\r", "").strip()

        special_tokens_to_remove = [
            "<|endoftext|>",
            "<|end|>",
            "<|return|>",
        ]

        for token in special_tokens_to_remove:
            text = text.replace(token, "").strip()

        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()

        stop_markers = [
            # GPT-OSS / Harmony-style markers
            "<|channel|>final<|message|>",
            "<|channel|>analysis<|message|>",
            "<|channel|>",
            "<|message|>",
            "<|end|>",
            "<|return|>",

            # Chat template markers
            "<|im_start|>",
            "<|im_end|>",
            "<|start_header_id|>",
            "<|end_header_id|>",

            # Role markers
            "\n_user",
            "_user",
            "\nUser:",
            "User:",
            "\nUSER:",
            "USER:",
            "\nAssistant:",
            "Assistant:",
            "\nASSISTANT:",
            "ASSISTANT:",

            # Prompt section markers
            "\nSelected Documents:",
            "\nRetrieved Documents:",
            "\nDocuments:",
            "\nQuestion:",
            "\nClaim:",
            "\nReturn JSON:",
            "\nOptions:",
            "\nAnswer:",
        ]

        cut_pos = None
        for marker in stop_markers:
            pos = text.find(marker)
            if pos != -1:
                cut_pos = pos if cut_pos is None else min(cut_pos, pos)

        if cut_pos is not None:
            text = text[:cut_pos].strip()

        prefix_patterns = [
            r"^\s*the\s+answer\s+is\s*:\s*",
            r"^\s*answer\s*:\s*",
            r"^\s*final\s+answer\s*:\s*",
            r"^\s*assistant\s*:\s*",
            r"^\s*final\s*:\s*",
        ]

        for pattern in prefix_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

        return strip_outer_quotes(text).strip()


SELF_RAG_JUDGE_SYS = (
    "You are a retrieval judge for Self-RAG. "
    "Given a question and retrieved documents, label each document as one of: "
    "relevant, irrelevant, or uncertain. "
    "Return strict JSON only. "
    "The JSON schema is: "
    '{"labels":[{"doc_id":1,"label":"relevant|irrelevant|uncertain","score":0}],'
    '"selected":[1]}. '
    "Prefer precision. Select all relevant documents and at most a few uncertain documents."
)

SELF_RAG_VERIFY_SYS = (
    "You are a verifier. "
    "Decide whether the proposed answer is fully supported by the selected documents. "
    "Return strict JSON only: "
    '{"supported": true, "need_more_docs": false}. '
    "supported=true only if the documents contain direct evidence."
)

FINAL_QA_SYS = (
    "You are a precise and concise question-answering system. "
    "If a passage is provided, rely mainly on it. "
    "Output ONLY the exact final answer. "
    "No explanations. No reasoning. No citations. No formatting."
)

FEVER_SYS = (
    "You are a fact-checking assistant. "
    "Determine whether the claim is supported by the provided context. "
    "Output ONLY 'SUPPORTS' or 'REFUTES'. "
    "Do not output anything else."
)

TRUTHFUL_MC_SYS = (
    "You are a multiple-choice QA system. "
    "Choose the single best option. "
    "Output ONLY the option letter, such as A, B, or C. "
    "Do not output explanations."
)


def safe_json_load(text: str) -> Optional[dict]:
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except Exception:
        try:
            import ast
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def build_docs_block(contexts: List[Any], top_k: int, doc_max_chars: int) -> str:
    lines = []

    for index, doc in enumerate(contexts[:top_k], start=1):
        content = truncate_text(extract_doc_text(doc), doc_max_chars)
        lines.append(f"[Document {index}] {content}")

    return "\n".join(lines)


def get_final_system_prompt(dataset_type: str) -> str:
    if dataset_type == "fever":
        return FEVER_SYS
    if dataset_type == "truthfulqa":
        return TRUTHFUL_MC_SYS
    return FINAL_QA_SYS


def get_truthful_options_block(
    query_id: str,
    choice_map: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    if query_id not in choice_map:
        raise ValueError(f"TruthfulQA query_id '{query_id}' is not found in the choice map.")

    choice_item = choice_map[query_id]
    choices = choice_item["choices"]

    option_lines = [
        f"{idx_to_letter(index)}. {choice}"
        for index, choice in enumerate(choices)
    ]

    return "\n".join(option_lines), choice_item


def selfrag_run_batch(
    generator: LocalGenerator,
    batch_items: List[Dict[str, Any]],
    top_k: int,
    select_m: int,
    verify: bool,
    judge_max_new_tokens: int,
    answer_max_new_tokens: int,
    verify_max_new_tokens: int,
    revise_max_new_tokens: int,
    doc_max_chars: int,
    dataset_type: str,
    truthful_choice_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[str], List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    judge_prompts = []

    for item in batch_items:
        question = item.get("query", item.get("question", ""))
        contexts = extract_contexts(item)
        docs_block = build_docs_block(contexts, top_k, doc_max_chars)

        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieved Documents:\n{docs_block}\n\n"
            f"Return JSON:"
        )

        judge_prompts.append(generator.build_chat_prompt(SELF_RAG_JUDGE_SYS, user_prompt))

    judge_outputs = generator.generate_batch(
        judge_prompts,
        max_new_tokens=judge_max_new_tokens,
    )

    selected_doc_ids_list = []
    judge_meta = []

    for output in judge_outputs:
        parsed = safe_json_load(output)
        selected = []
        labels = []

        if isinstance(parsed, dict):
            selected = parsed.get("selected", [])
            labels = parsed.get("labels", [])

        recovered = False

        if not isinstance(selected, list) or len(selected) == 0:
            if isinstance(labels, list):
                recovered_docs = []
                for label_item in labels:
                    if not isinstance(label_item, dict):
                        continue

                    doc_id = label_item.get("doc_id", None)
                    score = label_item.get("score", None)

                    try:
                        doc_id = int(doc_id)
                    except Exception:
                        continue

                    if isinstance(score, (int, float)) and score >= 2:
                        recovered_docs.append(doc_id)

                if recovered_docs:
                    selected = recovered_docs
                    recovered = True

        if not isinstance(selected, list) or len(selected) == 0:
            selected = list(range(1, min(select_m, top_k) + 1))

        selected_clean = []
        for doc_id in selected:
            try:
                doc_id = int(doc_id)
            except Exception:
                continue
            if 1 <= doc_id <= top_k:
                selected_clean.append(doc_id)

        selected_clean = selected_clean[:select_m]

        if not selected_clean:
            selected_clean = list(range(1, min(select_m, top_k) + 1))

        selected_doc_ids_list.append(selected_clean)
        judge_meta.append(
            {
                "judge_raw": output,
                "judge_parsed": parsed if isinstance(parsed, dict) else None,
                "selected": selected_clean,
                "selected_recovered_from_labels": recovered,
            }
        )

    final_system_prompt = get_final_system_prompt(dataset_type)
    answer_prompts = []
    truthful_meta_batch = [] if dataset_type == "truthfulqa" else None

    for item, selected_ids in zip(batch_items, selected_doc_ids_list):
        question = item.get("query", item.get("question", ""))
        contexts = extract_contexts(item)

        selected_lines = []
        for doc_id in selected_ids:
            if 1 <= doc_id <= len(contexts):
                content = truncate_text(extract_doc_text(contexts[doc_id - 1]), doc_max_chars)
                selected_lines.append(f"[Document {doc_id}] {content}")

        selected_docs_block = "\n".join(selected_lines)

        if dataset_type == "truthfulqa":
            if truthful_choice_map is None:
                raise ValueError("truthful_choice_map is required for dataset_type=truthfulqa.")

            query_id = str(item.get("query_id", item.get("id", ""))).strip()
            options_block, choice_item = get_truthful_options_block(query_id, truthful_choice_map)

            user_prompt = (
                f"Selected Documents:\n{selected_docs_block}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_block}\n\n"
                f"Answer:"
            )

            truthful_meta_batch.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "choices": choice_item["choices"],
                    "gold_idx": choice_item["gold_idx"],
                    "gold_texts": choice_item["gold_texts"],
                    "ground_truth": ensure_list_of_str(item.get("ground_truth", item.get("answers", []))),
                }
            )

        elif dataset_type == "fever":
            user_prompt = (
                f"Selected Documents:\n{selected_docs_block}\n\n"
                f"Claim: {question}\n\n"
                f"Answer:"
            )

        else:
            user_prompt = (
                f"Selected Documents:\n{selected_docs_block}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

        answer_prompts.append(generator.build_chat_prompt(final_system_prompt, user_prompt))

    answers = generator.generate_batch(
        answer_prompts,
        max_new_tokens=answer_max_new_tokens,
    )

    final_answers = answers[:]
    verify_meta = []

    if verify:
        verify_prompts = []

        for item, selected_ids, answer in zip(batch_items, selected_doc_ids_list, answers):
            question = item.get("query", item.get("question", ""))
            contexts = extract_contexts(item)

            selected_lines = []
            for doc_id in selected_ids:
                if 1 <= doc_id <= len(contexts):
                    content = truncate_text(extract_doc_text(contexts[doc_id - 1]), doc_max_chars)
                    selected_lines.append(f"[Document {doc_id}] {content}")

            selected_docs_block = "\n".join(selected_lines)

            user_prompt = (
                f"Question: {question}\n\n"
                f"Selected Documents:\n{selected_docs_block}\n\n"
                f"Proposed Answer: {answer}\n\n"
                f"Return JSON:"
            )

            verify_prompts.append(generator.build_chat_prompt(SELF_RAG_VERIFY_SYS, user_prompt))

        verify_outputs = generator.generate_batch(
            verify_prompts,
            max_new_tokens=verify_max_new_tokens,
        )

        revise_indices = []

        for index, output in enumerate(verify_outputs):
            parsed = safe_json_load(output) or {}
            supported = bool(parsed.get("supported", False))
            need_more_docs = bool(parsed.get("need_more_docs", False))

            verify_meta.append(
                {
                    "verify_raw": output,
                    "verify_parsed": parsed,
                }
            )

            if (not supported) and need_more_docs:
                revise_indices.append(index)

        if revise_indices:
            revise_prompts = []

            for index in revise_indices:
                item = batch_items[index]
                question = item.get("query", item.get("question", ""))
                contexts = extract_contexts(item)
                docs_block = build_docs_block(contexts, top_k, doc_max_chars)

                if dataset_type == "truthfulqa":
                    if truthful_choice_map is None:
                        raise ValueError("truthful_choice_map is required for dataset_type=truthfulqa.")

                    query_id = str(item.get("query_id", item.get("id", ""))).strip()
                    options_block, _ = get_truthful_options_block(query_id, truthful_choice_map)

                    user_prompt = (
                        f"Documents:\n{docs_block}\n\n"
                        f"Question: {question}\n\n"
                        f"Options:\n{options_block}\n\n"
                        f"Answer:"
                    )

                elif dataset_type == "fever":
                    user_prompt = (
                        f"Documents:\n{docs_block}\n\n"
                        f"Claim: {question}\n\n"
                        f"Answer:"
                    )

                else:
                    user_prompt = (
                        f"Documents:\n{docs_block}\n\n"
                        f"Question: {question}\n\n"
                        f"Answer:"
                    )

                revise_prompts.append(generator.build_chat_prompt(final_system_prompt, user_prompt))

            revised_answers = generator.generate_batch(
                revise_prompts,
                max_new_tokens=revise_max_new_tokens,
            )

            for local_index, global_index in enumerate(revise_indices):
                final_answers[global_index] = revised_answers[local_index]
                verify_meta[global_index]["revised"] = True
    else:
        verify_meta = [{} for _ in batch_items]

    merged_meta = []

    for judge_item, verify_item in zip(judge_meta, verify_meta):
        merged = {}
        merged.update(judge_item)
        merged.update(verify_item)
        merged_meta.append(merged)

    return final_answers, merged_meta, truthful_meta_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Run Self-RAG baseline.")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="results_selfrag.json")

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="general",
        choices=["general", "fever", "truthfulqa"],
    )

    parser.add_argument(
        "--truthfulqa_dev_jsonl",
        "--truthful_dev_jsonl",
        dest="truthfulqa_dev_jsonl",
        type=str,
        default=None,
    )

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--selfrag_select_m", type=int, default=3)
    parser.add_argument("--selfrag_verify", action="store_true")

    parser.add_argument("--judge_max_new_tokens", type=int, default=None)
    parser.add_argument("--answer_max_new_tokens", type=int, default=None)
    parser.add_argument("--verify_max_new_tokens", type=int, default=None)
    parser.add_argument("--revise_max_new_tokens", type=int, default=None)

    parser.add_argument("--doc_max_chars", type=int, default=1200)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--quantization_mode",
        type=str,
        default="none",
        choices=["none", "bnb4", "mxfp4"],
    )

    parser.add_argument("--debug_raw", action="store_true")
    parser.add_argument("--debug_n", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_type == "truthfulqa" and not args.truthfulqa_dev_jsonl:
        raise ValueError("--truthfulqa_dev_jsonl is required when --dataset_type truthfulqa.")

    if args.judge_max_new_tokens is None:
        if args.top_k >= 10:
            args.judge_max_new_tokens = 512
        elif args.top_k >= 5:
            args.judge_max_new_tokens = 256
        else:
            args.judge_max_new_tokens = 192

    if args.answer_max_new_tokens is None:
        args.answer_max_new_tokens = args.max_new_tokens

    if args.verify_max_new_tokens is None:
        args.verify_max_new_tokens = 128

    if args.revise_max_new_tokens is None:
        args.revise_max_new_tokens = args.answer_max_new_tokens

    print(f"Loading dataset from: {args.dataset_path}")
    data = load_data(args.dataset_path)

    for item in data:
        item["ground_truth"] = ensure_list_of_str(item.get("ground_truth", item.get("answers", [])))

    truthful_choice_map = None
    if args.dataset_type == "truthfulqa":
        print(f"Loading TruthfulQA choice map from: {args.truthfulqa_dev_jsonl}")
        truthful_choice_map = build_truthfulqa_choice_map(args.truthfulqa_dev_jsonl)
        print(f"Choice map size: {len(truthful_choice_map)}")

    generator = LocalGenerator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        load_in_4bit=args.load_in_4bit,
        quantization_mode=args.quantization_mode,
    )
    generator.debug_raw = args.debug_raw
    generator.debug_limit = args.debug_n

    print(
        "Starting Self-RAG | "
        f"dataset_type={args.dataset_type} | "
        f"top_k={args.top_k} | "
        f"select_m={args.selfrag_select_m} | "
        f"verify={args.selfrag_verify}"
    )

    all_predictions = []
    all_selfrag_meta = []
    truthful_meta_all = [] if args.dataset_type == "truthfulqa" else None

    for start in tqdm(range(0, len(data), args.batch_size), desc="Self-RAG"):
        batch_items = data[start:start + args.batch_size]

        predictions, selfrag_meta, truthful_meta = selfrag_run_batch(
            generator=generator,
            batch_items=batch_items,
            top_k=args.top_k,
            select_m=args.selfrag_select_m,
            verify=args.selfrag_verify,
            judge_max_new_tokens=args.judge_max_new_tokens,
            answer_max_new_tokens=args.answer_max_new_tokens,
            verify_max_new_tokens=args.verify_max_new_tokens,
            revise_max_new_tokens=args.revise_max_new_tokens,
            doc_max_chars=args.doc_max_chars,
            dataset_type=args.dataset_type,
            truthful_choice_map=truthful_choice_map,
        )

        all_predictions.extend(predictions)
        all_selfrag_meta.extend(selfrag_meta)

        if args.dataset_type == "truthfulqa":
            assert truthful_meta_all is not None
            assert truthful_meta is not None
            truthful_meta_all.extend(truthful_meta)

    report = {
        "config": vars(args),
        "metrics": {},
        "details": [],
    }

    if args.dataset_type == "truthfulqa":
        assert truthful_meta_all is not None

        acc, acc_details = compute_truthfulqa_acc(all_predictions, truthful_meta_all)
        report["metrics"] = {"acc": float(acc)}

        for detail, selfrag_meta in zip(acc_details, all_selfrag_meta):
            detail["selfrag_meta"] = selfrag_meta
            report["details"].append(detail)

    else:
        golds = [item["ground_truth"] for item in data]

        if args.dataset_type == "fever":
            metrics = calculate_fever_metrics(all_predictions, golds)
        else:
            metrics = calculate_qa_metrics(all_predictions, golds)

        recall = calculate_recall(data, top_k=args.top_k)

        report["metrics"] = {
            **metrics,
            "recall": float(recall),
        }

        for item, prediction, selfrag_meta in zip(data, all_predictions, all_selfrag_meta):
            gold_list = item.get("ground_truth", [])
            report["details"].append(
                {
                    "query_id": item.get("query_id", item.get("id", "")),
                    "question": item.get("query", item.get("question", "")),
                    "ground_truth": gold_list,
                    "prediction": prediction,
                    "selfrag_meta": selfrag_meta,
                }
            )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Self-RAG Report")
    print("-" * 60)
    print(f"dataset_type: {args.dataset_type}")
    for key, value in report["metrics"].items():
        print(f"{key}: {value:.4f}")
    print(f"saved to: {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()