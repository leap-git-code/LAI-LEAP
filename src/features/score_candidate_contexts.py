#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import re
import string
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig

try:
    from transformers import Mxfp4Config
except ImportError:
    Mxfp4Config = None


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    return question[0].lower() + question[1:]


def ensure_list_of_str(x):
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, dict):
        for k in ["answers", "gold_answers", "golden_answers", "ground_truth", "text", "answer"]:
            if k in x:
                return ensure_list_of_str(x[k])
    s = str(x).strip()
    return [s] if s else []


def load_data(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".json"):
            data = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def idx_to_letter(i: int) -> str:
    return chr(ord("A") + i)


def build_truthfulqa_choice_map(dev_jsonl_path: str):
    rows = load_data(dev_jsonl_path)
    choice_map = {}

    def clean_text(s):
        return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip()

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

        choices = [clean_text(c) for c in choices if clean_text(c)]
        if len(choices) == 0:
            continue

        metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
        incorrect = metadata.get("incorrect_answers", [])
        if not isinstance(incorrect, list):
            incorrect = []
        incorrect = [clean_text(x) for x in incorrect if clean_text(x)]

        gold_idx_clean = []
        for x in gold_idx:
            try:
                gold_idx_clean.append(int(x))
            except Exception:
                pass
        if len(gold_idx_clean) == 0:
            gold_idx_clean = [0]

        if len(choices) <= 1 and len(incorrect) > 0:
            gold_text = choices[0]
            merged = [gold_text]
            seen = {gold_text}
            for wrong_answer in incorrect:
                if wrong_answer not in seen:
                    merged.append(wrong_answer)
                    seen.add(wrong_answer)
            choices_all = merged
            gold_idx_clean = [0]
            gold_texts = [gold_text]
        else:
            choices_all = []
            seen = set()
            for choice in choices:
                if choice not in seen:
                    choices_all.append(choice)
                    seen.add(choice)
            for wrong_answer in incorrect:
                if wrong_answer not in seen:
                    choices_all.append(wrong_answer)
                    seen.add(wrong_answer)

            gold_texts = []
            for gi in gold_idx_clean:
                if 0 <= gi < len(choices_all):
                    gold_texts.append(choices_all[gi])

        choice_map[key] = {
            "choices": choices_all,
            "gold_idx": gold_idx_clean,
            "gold_texts": gold_texts,
        }

    return choice_map


def build_user_message_content_truthfulqa(question: str, choices, context_text: str = None) -> str:
    def one_line(s):
        return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip()

    candidate_block = "\n".join([one_line(c) for c in choices])
    q = one_line(question)

    if context_text and len(str(context_text).strip()) > 0:
        context_text = str(context_text).strip()
        return (
            f"Background:\n{context_text}\n\n"
            f"Question: {q}\n\n"
            f"Candidate Answers (choose ONE and output it verbatim):\n"
            f"{candidate_block}\n\n"
            f"Answer:"
        )

    return (
        f"Question: {q}\n\n"
        f"Candidate Answers (choose ONE and output it verbatim):\n"
        f"{candidate_block}\n\n"
        f"Answer:"
    )


def is_invalid_token(token_str: str) -> bool:
    s = token_str.strip()
    if s == "":
        return True
    if s.lower() in {"the", "a", "an"}:
        return True
    if s in string.punctuation:
        return True
    if s == "<|eot_id|>":
        return True
    return False


def generate_case_space_variants(text: str):
    base = text.strip()
    variants = {base, base.lower(), base.upper(), base.swapcase()}

    no_articles = re.sub(r"\b(the|a|an)\b", " ", base, flags=re.IGNORECASE)
    no_articles = " ".join(no_articles.split())
    variants.add(no_articles)

    no_punc = "".join(ch for ch in base if ch not in string.punctuation)
    variants.add(no_punc)

    final_variants = set()
    for v in variants:
        if not v.strip():
            continue
        final_variants.add(v)
        final_variants.add(v.replace(" ", ""))
        final_variants.add(v.lower())

    return final_variants


def layer_weight(layer_id, hit_layer, alpha=1.0):
    return 1 / (1 + math.exp(-alpha * (layer_id - hit_layer)))


def compute_step_score(per_layer_topk, target_token_ids_set, alpha=1.0):
    num_layers = len(per_layer_topk)
    hit_layer = None

    for layer_id in range(num_layers):
        for item in per_layer_topk[layer_id]:
            if item["token_id"] in target_token_ids_set:
                hit_layer = layer_id
                break
        if hit_layer is not None:
            break

    if hit_layer is None:
        hit_layer = num_layers

    token_score = 0.0
    for layer_id in range(num_layers):
        matched_logit = 0.0
        for item in per_layer_topk[layer_id]:
            if item["token_id"] in target_token_ids_set:
                matched_logit = item["logit"]
                break
        weight = layer_weight(layer_id, hit_layer, alpha=alpha)
        token_score += weight * matched_logit

    return token_score


def get_answer_token_ids_context_aware(tokenizer, prompt: str, answers_list):
    answer_token_map = {}
    all_valid_ids = set()

    base_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    base_len = len(base_ids)

    for ans_text in answers_list:
        raw_answer = str(ans_text).strip()
        if not raw_answer:
            continue

        text_variants = generate_case_space_variants(raw_answer)
        merged_ids = []

        for variant in text_variants:
            variant = str(variant).strip()
            if not variant:
                continue

            full_text = prompt + variant
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

            if len(full_ids) <= base_len:
                continue

            ans_ids = full_ids[base_len:]
            merged_ids.extend(int(t) for t in ans_ids)

        seen = set()
        unique_ids = []
        for tid in merged_ids:
            if tid not in seen:
                seen.add(tid)
                unique_ids.append(tid)

        token_details = [
            {"id": int(tid), "token": tokenizer.convert_ids_to_tokens(int(tid))}
            for tid in unique_ids
        ]

        answer_token_map[raw_answer] = token_details
        all_valid_ids.update(unique_ids)

    return answer_token_map, all_valid_ids


def build_user_message_content_qa(question: str, context_text: str = None) -> str:
    q = normalize_question(question)
    if context_text and len(context_text.strip()) > 0:
        return f"Context: {context_text}\nQuestion: {q}\nAnswer:(Please output ONLY the answer)"
    return f"Question: {q}\nAnswer:(Please output ONLY the answer)"


class ModelWrapper:
    def __init__(
        self,
        model_path: str,
        device_id: int,
        load_in_4bit: bool = False,
        quantization_mode: str = "none",
    ):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.harmony_final_prefix = "<|channel|>final<|message|>"

        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = (getattr(cfg, "model_type", "") or "").lower()
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
                    "Mxfp4Config is not available. Please upgrade transformers "
                    "or use --quantization_mode none/bnb4."
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
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()

        self.lm_head_weight = self.model.lm_head.weight.detach()
        self.hooks_container = {}
        self.hook_handles = []

    def response_generation(self, full_answer: str) -> str:
        if full_answer is None:
            return ""

        text = str(full_answer).strip()

        if "assistant" in text:
            text = text.split("assistant\n", 1)[-1].lstrip(":").strip()

        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        stop_markers = [
            "_user", "_assistant", "\nuser", "._user", "._",
            "Context:", "Question:", "Answer:",
            "<|im_start|>", "<|im_end|>",
            "<|user|>", "<|assistant|>", "<|system|>",
            "\n",
        ]

        cut_pos = len(text)
        for marker in stop_markers:
            pos = text.find(marker)
            if pos != -1:
                cut_pos = min(cut_pos, pos)

        text = text[:cut_pos].strip()
        text = text.replace("\r", "").replace("\t", "").strip()

        return text

    def _apply_template(self, system: str, user: str) -> str:
        messages = []

        if self.model_family == "gpt" and system and len(system) > 0:
            messages.append({"role": "developer", "content": system})
        elif system and len(system) > 0:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": user})

        if self.model_family == "qwen":
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        elif self.model_family == "gpt":
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
        else:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if self.model_family == "gpt":
            rendered += self.harmony_final_prefix

        return rendered

    def register_hooks(self):
        self.remove_hooks()
        self.hooks_container = {}

        def make_hook(layer_id):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                self.hooks_container[layer_id] = out.detach()
            return hook

        layers = getattr(self.model, "model", self.model).layers
        for i, layer in enumerate(layers):
            self.hook_handles.append(layer.register_forward_hook(make_hook(i)))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.hooks_container.clear()


def run_generation_trace_batch(
    wrapper: ModelWrapper,
    batch_prompts,
    batch_answers_list,
    max_new_tokens: int,
    top_k_lens: int,
    temperature: float = 0.1,
    top_p: float = 0.9,
    do_sample: bool = False,
    dataset_type="default",
):
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    device = wrapper.device

    batch_size = len(batch_prompts)
    assert batch_size == len(batch_answers_list)

    batch_target_sets = []
    batch_answers_maps = []
    for i in range(batch_size):
        answer_map, valid_ids = get_answer_token_ids_context_aware(
            tokenizer,
            batch_prompts[i],
            batch_answers_list[i],
        )
        batch_target_sets.append(valid_ids)
        batch_answers_maps.append(answer_map)

    enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    wrapper.register_hooks()

    batch_step_records = [[] for _ in range(batch_size)]
    batch_generated_ids = [[] for _ in range(batch_size)]
    batch_valid_step_scores = [[] for _ in range(batch_size)]
    batch_select_num = [0 for _ in range(batch_size)]
    batch_generated_token_num = [0 for _ in range(batch_size)]
    batch_finished = [False for _ in range(batch_size)]
    batch_finished_reason = ["max_length" for _ in range(batch_size)]

    past_key_values = None
    curr_input_ids = input_ids

    lm_head_weight_t = wrapper.lm_head_weight.T

    with torch.inference_mode():
        for step in range(max_new_tokens):
            if all(batch_finished):
                break

            for b in range(batch_size):
                if not batch_finished[b]:
                    batch_generated_token_num[b] += 1

            wrapper.hooks_container.clear()

            outputs = model(
                input_ids=curr_input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample:
                logits_to_sample = next_token_logits / (temperature if temperature > 0 else 1.0)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_to_sample, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1,
                        sorted_indices,
                        sorted_indices_to_remove,
                    )
                    logits_to_sample = logits_to_sample.masked_fill(indices_to_remove, -float("inf"))
                probs = F.softmax(logits_to_sample, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            layer_ids = sorted(wrapper.hooks_container.keys())
            per_layer_topk_by_sample = [[] for _ in range(batch_size)]

            for layer_id in layer_ids:
                hidden_state = wrapper.hooks_container[layer_id][:, -1, :]
                layer_logits = torch.matmul(hidden_state, lm_head_weight_t)
                values, indices = torch.topk(layer_logits, k=top_k_lens, dim=-1)

                indices_cpu = indices.to("cpu").tolist()
                values_cpu = values.float().to("cpu").tolist()

                for b in range(batch_size):
                    if batch_finished[b]:
                        continue

                    layer_result = []
                    for k in range(top_k_lens):
                        token_id = int(indices_cpu[b][k])
                        logit_value = float(values_cpu[b][k])
                        layer_result.append(
                            {
                                "token": tokenizer.decode([token_id]),
                                "token_id": token_id,
                                "logit": logit_value,
                            }
                        )
                    per_layer_topk_by_sample[b].append(layer_result)

            for b in range(batch_size):
                if batch_finished[b]:
                    continue

                token_id = int(next_tokens[b].item())
                batch_generated_ids[b].append(token_id)

                if token_id == tokenizer.eos_token_id:
                    batch_finished[b] = True
                    batch_finished_reason[b] = "eos"
                    batch_step_records[b].append(
                        {
                            "step": step,
                            "token_id": token_id,
                            "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                            "step_score": -1,
                            "layers_lens": [],
                        }
                    )
                    continue

                generated_so_far = tokenizer.decode(batch_generated_ids[b], skip_special_tokens=False)

                stop_markers = [
                    "\n._user", "._user", "\n_user", "_user", "._",
                    "\nContext:", "Context:",
                    "\nQuestion:", "Question:",
                    "\nAnswer:", "Answer:",
                    "<|im_start|>", "<|im_end|>",
                    "<|assistant|>", "<|user|>", "<|system|>",
                    "_assistant", "\n",
                ]

                cut_pos = None
                for marker in stop_markers:
                    pos = generated_so_far.find(marker)
                    if pos != -1:
                        cut_pos = pos if cut_pos is None else min(cut_pos, pos)

                if cut_pos is not None:
                    batch_finished[b] = True
                    batch_finished_reason[b] = "chat_stop_text"

                    truncated_text = generated_so_far[:cut_pos]
                    truncated_ids = tokenizer(
                        truncated_text,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids[0].tolist()
                    batch_generated_ids[b] = truncated_ids

                    batch_step_records[b].append(
                        {
                            "step": step,
                            "token_id": token_id,
                            "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                            "step_score": -1,
                            "layers_lens": [],
                        }
                    )
                    continue

                token_str = tokenizer.decode([token_id], skip_special_tokens=False)

                if is_invalid_token(token_str):
                    step_score = -1
                    batch_step_records[b].append(
                        {
                            "step": step,
                            "token_id": token_id,
                            "token_text": token_str,
                            "step_score": step_score,
                            "layers_lens": [],
                        }
                    )
                else:
                    per_layer_topk_list = per_layer_topk_by_sample[b]
                    step_score = compute_step_score(per_layer_topk_list, batch_target_sets[b])

                    batch_valid_step_scores[b].append(step_score)
                    batch_select_num[b] += 1

                    batch_step_records[b].append(
                        {
                            "step": step,
                            "token_id": token_id,
                            "token_text": token_str,
                            "step_score": float(step_score),
                            "layers_lens": per_layer_topk_list,
                        }
                    )

            curr_input_ids = next_tokens.unsqueeze(-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )

    wrapper.remove_hooks()

    results = []
    for b in range(batch_size):
        prompt_text = batch_prompts[b]
        answers_list = batch_answers_list[b]

        answer_token_lengths = []
        base_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        base_len = base_ids.shape[0]

        for ans_text in answers_list:
            raw_ans = str(ans_text).strip()
            full_ids = tokenizer(
                prompt_text + " " + raw_ans,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            ans_len = full_ids.shape[0] - base_len
            if ans_len <= 0:
                ans_len = 1
            answer_token_lengths.append(ans_len)

        if len(answer_token_lengths) == 0:
            min_answer_token_len = 1
        else:
            min_answer_token_len = min(answer_token_lengths)

        select_num = batch_select_num[b]
        generated_token_num = batch_generated_token_num[b]
        valid_scores = batch_valid_step_scores[b]

        if select_num == 0:
            final_score = 0.0
        else:
            if min_answer_token_len > generated_token_num + 2:
                final_score = float(sum(valid_scores) / min_answer_token_len)
            else:
                final_score = float(sum(valid_scores) / select_num)

        generated_text = tokenizer.decode(batch_generated_ids[b], skip_special_tokens=True)
        final_answer = wrapper.response_generation(generated_text)
        final_answer = final_answer.replace("\n", "").strip()

        results.append(
            {
                "prompt": prompt_text,
                "generated_text": generated_text,
                "final_answer": final_answer,
                "final_score": final_score,
                "finished_reason": batch_finished_reason[b],
                "select_num": select_num,
                "G_token_num": generated_token_num,
                "K_token_min": min_answer_token_len,
                "target_answers_map": batch_answers_maps[b],
                "steps": batch_step_records[b],
            }
        )

    return results


def extract_question(item: dict) -> str:
    return item.get("question") or item.get("query") or ""


def extract_answers(item: dict):
    answers = item.get("answers", None)
    if answers is None:
        answers = item.get("ground_truth", None)
    if answers is None:
        answers = []
    if not isinstance(answers, list):
        answers = [answers]
    if len(answers) > 0 and isinstance(answers[0], list):
        answers = answers[0]
    return answers


def extract_query_id(item: dict) -> str:
    query_id = item.get("query_id", None)
    if query_id is None:
        query_id = item.get("id", None)
    if query_id is None:
        query_id = "unknown"
    return str(query_id)


def extract_candidates(item: dict, top_k: int):
    if "candidates" in item and isinstance(item["candidates"], list):
        candidates = item["candidates"]
        output = []
        for candidate in candidates[:top_k]:
            if isinstance(candidate, str):
                output.append({"text": candidate})
            elif isinstance(candidate, dict):
                output.append(candidate)
        return output

    if "ctxs" in item and isinstance(item["ctxs"], list):
        return item["ctxs"][:top_k]

    if "selected_contexts" in item and isinstance(item["selected_contexts"], list):
        output = []
        for text in item["selected_contexts"][:top_k]:
            output.append({"text": text})
        return output

    return []


def extract_context_text(candidate: dict) -> str:
    if isinstance(candidate, str):
        return candidate
    return candidate.get("contents") or candidate.get("text") or candidate.get("context") or ""


def load_input_dataset(input_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        if input_file.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def worker_process(
    gpu_id,
    dataset,
    model_path,
    output_dir,
    top_k,
    batch_size,
    load_in_4bit,
    dataset_type="default",
    truthfulqa_dev_jsonl=None,
    truthfulqa_top_k_ctx=5,
    start_item_idx=0,
    quantization_mode="none",
):
    wrapper = ModelWrapper(
        model_path,
        gpu_id,
        load_in_4bit=load_in_4bit,
        quantization_mode=quantization_mode,
    )

    system_prompt_qa = (
        "You are a precise and concise question-answering system. "
        "If a passage is provided, you should rely mainly on it; "
        "otherwise, use your own knowledge. "
        "Output ONLY the exact final answer as a SINGLE LINE of plain text. "
        "Absolutely NO explanations. NO reasoning. "
        "NO restating the question. NO extra text. NO quotes. NO formatting. "
        "NO leading or trailing spaces. "
        "Do NOT include any newline characters or line breaks. "
    )

    system_prompt_fever = (
        "You are a fact-checking assistant. "
        "Determine whether the claim is supported by the provided context. "
        "Output ONLY 'SUPPORTS' or 'REFUTES'. "
        "Do not output anything else."
    )

    system_prompt_truthfulqa = (
        "You are a precise selector. "
        "You will be given a question, background, and several candidate answers. "
        "Choose the single best candidate. "
        "Output ONLY the exact text of ONE candidate answer, copied verbatim. "
        "Do NOT output any label, letter, number, index, or bullet. "
        "Do NOT output multiple candidates. "
        "Output as a single line."
    )

    choice_map = None
    if dataset_type == "truthfulqa":
        if not truthfulqa_dev_jsonl:
            raise ValueError("dataset_type=truthfulqa requires --truthfulqa_dev_jsonl.")
        choice_map = build_truthfulqa_choice_map(truthfulqa_dev_jsonl)

    os.makedirs(output_dir, exist_ok=True)

    for item_idx, item in enumerate(tqdm(dataset, desc="Processing")):
        if item_idx < start_item_idx:
            continue

        query_id = extract_query_id(item)
        question = extract_question(item)
        answers = extract_answers(item)

        candidates = extract_candidates(item, top_k=top_k)
        if not candidates:
            continue

        safe_qid = query_id.replace("/", "_")

        for start in range(0, len(candidates), batch_size):
            candidate_batch = candidates[start:start + batch_size]

            batch_prompts = []
            batch_answers = []
            batch_metas = []

            for idx_in_batch, candidate in enumerate(candidate_batch):
                context_text = extract_context_text(candidate)

                if dataset_type == "truthfulqa":
                    if query_id not in choice_map:
                        continue
                    choices = choice_map[query_id]["choices"]
                    user_content = build_user_message_content_truthfulqa(question, choices, context_text)
                    full_prompt = wrapper._apply_template(system_prompt_truthfulqa, user_content)

                elif dataset_type == "fever":
                    user_content = build_user_message_content_qa(question, context_text)
                    full_prompt = wrapper._apply_template(system_prompt_fever, user_content)

                else:
                    user_content = build_user_message_content_qa(question, context_text)
                    full_prompt = wrapper._apply_template(system_prompt_qa, user_content)

                trace_answers = answers

                batch_prompts.append(full_prompt)
                batch_answers.append(trace_answers)

                rank_index = start + idx_in_batch
                batch_metas.append(
                    {
                        "rank_index": rank_index,
                        "text": context_text,
                    }
                )

            if not batch_prompts:
                continue

            batch_results = run_generation_trace_batch(
                wrapper=wrapper,
                batch_prompts=batch_prompts,
                batch_answers_list=batch_answers,
                max_new_tokens=64,
                top_k_lens=10,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                dataset_type=dataset_type,
            )

            for i, result in enumerate(batch_results):
                meta = batch_metas[i]
                output_item = {
                    "query_id": query_id,
                    "rank_index": meta["rank_index"],
                    "question": question,
                    "text": meta["text"],
                    "ground_truth": answers,
                    **result,
                }

                file_name = f"{safe_qid}_{meta['rank_index']}.json"
                file_path = os.path.join(output_dir, file_name)

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(output_item, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True, help="Input JSON or JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for candidate scores.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or HuggingFace name of the language model.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5, help="Candidate batch size per query.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Backward-compatible alias for --quantization_mode bnb4.")

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="default",
        choices=["default", "fever", "truthfulqa"],
        help="Dataset type for task-specific prompting.",
    )

    parser.add_argument(
        "--truthfulqa_dev_jsonl",
        type=str,
        default=None,
        help="TruthfulQA dev file containing choices and golden_answers.",
    )

    parser.add_argument(
        "--truthfulqa_top_k_ctx",
        type=int,
        default=5,
        help="Reserved option for TruthfulQA context construction.",
    )

    parser.add_argument(
        "--start_item_idx",
        type=int,
        default=0,
        help="Resume from a given dataset index.",
    )

    parser.add_argument(
        "--quantization_mode",
        type=str,
        default="none",
        choices=["none", "bnb4", "mxfp4"],
        help="Quantization mode for loading large language models.",
    )

    args = parser.parse_args()
    dataset = load_input_dataset(args.input_file)

    worker_process(
        gpu_id=args.gpu_id,
        dataset=dataset,
        model_path=args.model_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
        dataset_type=args.dataset_type,
        truthfulqa_dev_jsonl=args.truthfulqa_dev_jsonl,
        truthfulqa_top_k_ctx=args.truthfulqa_top_k_ctx,
        start_item_idx=args.start_item_idx,
        quantization_mode=args.quantization_mode,
    )

    print("Done.")


if __name__ == "__main__":
    main()