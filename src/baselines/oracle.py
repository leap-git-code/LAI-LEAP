#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .llm_utils import LocalCausalLM
    from .utils import (
        extract_candidates,
        extract_ground_truth,
        extract_query,
        extract_query_id,
        get_candidate_text,
        read_json_or_jsonl,
        write_jsonl_line,
    )
except ImportError:
    from llm_utils import LocalCausalLM
    from utils import (
        extract_candidates,
        extract_ground_truth,
        extract_query,
        extract_query_id,
        get_candidate_text,
        read_json_or_jsonl,
        write_jsonl_line,
    )


SYSTEM_PROMPT_GENERAL = (
    "You are a precise and concise question-answering system. "
    "If a passage is provided, you should rely mainly on it; "
    "otherwise, use your own knowledge. "
    "Output ONLY the exact final answer. Absolutely NO explanations. NO reasoning. "
    "NO restating the question. NO extra text. NO quotes. NO formatting. "
    "NO leading/trailing words. Output ONLY the answer content itself."
)

SYSTEM_PROMPT_FEVER = (
    "You are a fact-checking assistant. "
    "Determine whether the claim is supported by the provided context. "
    "Output ONLY 'SUPPORTS' or 'REFUTES'. "
    "Do not output anything else."
)

SYSTEM_PROMPT_TRUTHFULQA = (
    "You are a multiple-choice QA system. "
    "Choose the single best option. "
    "Output ONLY the option letter, such as A, B, or C. "
    "Do not output explanations."
)


GENERAL_DATASET_TYPES = {
    "general",
    "default",
    "nq",
    "hotpotqa",
    "hotpotnq",
    "2wiki",
    "2wikimultihopqa",
    "musique",
    "medical",
    "covidqa",
    "finance",
}


def ensure_list_of_str(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, dict):
        for key in [
            "answers",
            "gold_answers",
            "golden_answers",
            "ground_truth",
            "text",
            "answer",
        ]:
            if key in value:
                return ensure_list_of_str(value[key])
        return [json.dumps(value, ensure_ascii=False)]

    raw_values = value if isinstance(value, list) else [value]
    output = []

    for item in raw_values:
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


def pick_first_ground_truth(item: Dict[str, Any]) -> str:
    answers = extract_ground_truth(item)
    answers = ensure_list_of_str(answers)
    return answers[0] if answers else ""


def idx_to_letter(index: int) -> str:
    if index < 0:
        return ""
    if index < 26:
        return chr(ord("A") + index)
    return f"OPT{index}"


def build_truthfulqa_choice_map(dev_jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    choice_map: Dict[str, Dict[str, Any]] = {}

    for item in read_json_or_jsonl(dev_jsonl_path):
        query_id = str(item.get("id", item.get("query_id", ""))).strip()
        if not query_id:
            continue

        choices = item.get("choices", None)
        golden_answers = item.get("golden_answers", None)

        if not isinstance(choices, list) or len(choices) == 0:
            continue

        choices = [str(choice).strip() for choice in choices if str(choice).strip()]

        if not isinstance(golden_answers, list):
            golden_answers = []

        gold_idx = []
        for value in golden_answers:
            try:
                index = int(value)
                if 0 <= index < len(choices):
                    gold_idx.append(index)
            except Exception:
                pass

        gold_texts = [choices[index] for index in gold_idx]

        choice_map[query_id] = {
            "choices": choices,
            "gold_idx": gold_idx,
            "gold_texts": gold_texts,
        }

    return choice_map


def normalize_dataset_type(dataset_type: str) -> str:
    dataset_type = (dataset_type or "general").lower()

    if dataset_type == "fever":
        return "fever"

    if dataset_type == "truthfulqa":
        return "truthfulqa"

    if dataset_type in GENERAL_DATASET_TYPES:
        return "general"

    return "general"


def pick_system_prompt(dataset_type: str) -> str:
    normalized_type = normalize_dataset_type(dataset_type)

    if normalized_type == "fever":
        return SYSTEM_PROMPT_FEVER

    if normalized_type == "truthfulqa":
        return SYSTEM_PROMPT_TRUTHFULQA

    return SYSTEM_PROMPT_GENERAL


def build_user_message_general(context: str, query: str) -> str:
    return f"Passage:\n{context}\n\nQuestion: {query}\nAnswer:"


def build_user_message_fever(context: str, claim: str) -> str:
    return f"Context:\n{context}\n\nClaim: {claim}\nAnswer:"


def build_user_message_truthfulqa(context: str, question: str, choices: List[str]) -> str:
    options_block = "\n".join(
        f"{idx_to_letter(index)}. {choice}"
        for index, choice in enumerate(choices)
    )

    if context and context.strip():
        return (
            f"Background:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_block}\n\n"
            f"Answer:"
        )

    return (
        f"Question: {question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Answer:"
    )


def build_user_message(
    dataset_type: str,
    context: str,
    query: str,
    choices: Optional[List[str]] = None,
) -> str:
    normalized_type = normalize_dataset_type(dataset_type)

    if normalized_type == "fever":
        return build_user_message_fever(context, query)

    if normalized_type == "truthfulqa":
        return build_user_message_truthfulqa(context, query, choices or [])

    return build_user_message_general(context, query)


def make_example_tensors(
    generator: LocalCausalLM,
    dataset_type: str,
    query: str,
    context: str,
    answer: str,
    max_length: int,
    append_eos: bool,
    choices: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    tokenizer = generator.tokenizer

    answer_text = str(answer)

    if append_eos and tokenizer.eos_token and not answer_text.endswith(tokenizer.eos_token):
        answer_text = answer_text + tokenizer.eos_token

    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

    system_prompt = pick_system_prompt(dataset_type)
    user_content = build_user_message(
        dataset_type=dataset_type,
        context=context,
        query=query,
        choices=choices,
    )

    prompt_text = generator.apply_chat_template(system_prompt, user_content)
    context_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    total_length = len(context_ids) + len(answer_ids)
    if total_length > max_length:
        keep_context_length = max_length - len(answer_ids)
        if keep_context_length <= 0:
            context_ids = []
        else:
            context_ids = context_ids[-keep_context_length:]

    input_ids = context_ids + answer_ids
    labels = [-100] * len(context_ids) + answer_ids

    return input_ids, labels


def pad_batch_right(
    input_ids_list: List[List[int]],
    labels_list: List[List[int]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(input_ids_list)
    max_length = max(len(input_ids) for input_ids in input_ids_list) if batch_size else 0

    input_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

    for index, (input_ids_item, labels_item) in enumerate(zip(input_ids_list, labels_list)):
        length = len(input_ids_item)
        input_ids[index, :length] = torch.tensor(input_ids_item, dtype=torch.long)
        labels[index, :length] = torch.tensor(labels_item, dtype=torch.long)
        attention_mask[index, :length] = 1

    return input_ids, labels, attention_mask


@torch.inference_mode()
def per_example_nll_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_size, seq_len = shift_labels.shape
    vocab_size = shift_logits.size(-1)

    token_loss = F.cross_entropy(
        shift_logits.view(batch_size * seq_len, vocab_size),
        shift_labels.view(batch_size * seq_len),
        reduction="none",
        ignore_index=-100,
    ).view(batch_size, seq_len)

    valid_mask = (shift_labels != -100).to(token_loss.dtype)
    denom = valid_mask.sum(dim=1).clamp_min(1.0)
    loss_sum = (token_loss * valid_mask).sum(dim=1)

    return loss_sum / denom


def normalize_fever_answer(answer: str) -> str:
    answer = str(answer).strip().upper()

    if answer in {"SUPPORTS", "REFUTES"}:
        return answer

    if "SUPPORT" in answer:
        return "SUPPORTS"

    if "REFUTE" in answer:
        return "REFUTES"

    return ""


def fallback_topk(docs: List[str], top_k_output: int) -> List[str]:
    selected = docs[:top_k_output]
    if len(selected) < top_k_output:
        selected += [""] * (top_k_output - len(selected))
    return selected


def oracle_rank_topk(
    generator: LocalCausalLM,
    dataset_type: str,
    query_id: str,
    query: str,
    answer: str,
    candidates: List[Any],
    top_k_input: int,
    top_k_output: int,
    batch_size: int,
    max_length: int,
    append_eos: bool,
    choice_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    tokenizer = generator.tokenizer
    model = generator.model
    device = next(model.parameters()).device

    normalized_type = normalize_dataset_type(dataset_type)

    candidate_pool = candidates[:top_k_input]
    docs = [get_candidate_text(candidate) for candidate in candidate_pool]
    docs = [doc for doc in docs if isinstance(doc, str)]

    if not docs:
        return [""] * top_k_output

    choices = None
    gold_letters = None

    if normalized_type == "truthfulqa":
        if not choice_map or query_id not in choice_map:
            return fallback_topk(docs, top_k_output)

        choices = choice_map[query_id].get("choices", [])
        gold_idx = choice_map[query_id].get("gold_idx", [])
        gold_letters = [
            idx_to_letter(int(index))
            for index in gold_idx
            if str(index).isdigit()
        ]
        gold_letters = [letter for letter in gold_letters if letter]

        if not gold_letters:
            return fallback_topk(docs, top_k_output)

    if normalized_type != "truthfulqa" and not str(answer).strip():
        return fallback_topk(docs, top_k_output)

    if normalized_type == "truthfulqa":
        min_losses = torch.full((len(docs),), float("inf"), dtype=torch.float32)

        assert gold_letters is not None

        for gold_letter in gold_letters:
            losses = []

            for start in range(0, len(docs), batch_size):
                batch_docs = docs[start:start + batch_size]
                input_ids_list = []
                labels_list = []

                for doc in batch_docs:
                    input_ids, labels = make_example_tensors(
                        generator=generator,
                        dataset_type=dataset_type,
                        query=query,
                        context=doc,
                        answer=gold_letter,
                        max_length=max_length,
                        append_eos=append_eos,
                        choices=choices,
                    )
                    input_ids_list.append(input_ids)
                    labels_list.append(labels)

                input_ids, labels, attention_mask = pad_batch_right(
                    input_ids_list=input_ids_list,
                    labels_list=labels_list,
                    pad_id=tokenizer.pad_token_id,
                )

                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)

                batch_loss = per_example_nll_loss(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                losses.extend(batch_loss.float().detach().cpu().tolist())

            loss_tensor = torch.tensor(losses, dtype=torch.float32)
            min_losses = torch.minimum(min_losses, loss_tensor)

        scores = (-min_losses).tolist()

    else:
        scores = []

        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start:start + batch_size]
            input_ids_list = []
            labels_list = []

            for doc in batch_docs:
                input_ids, labels = make_example_tensors(
                    generator=generator,
                    dataset_type=dataset_type,
                    query=query,
                    context=doc,
                    answer=answer,
                    max_length=max_length,
                    append_eos=append_eos,
                    choices=None,
                )
                input_ids_list.append(input_ids)
                labels_list.append(labels)

            input_ids, labels, attention_mask = pad_batch_right(
                input_ids_list=input_ids_list,
                labels_list=labels_list,
                pad_id=tokenizer.pad_token_id,
            )

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            batch_loss = per_example_nll_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            scores.extend((-batch_loss).float().detach().cpu().tolist())

    sorted_indices = sorted(
        range(len(docs)),
        key=lambda index: scores[index],
        reverse=True,
    )

    selected = [docs[index] for index in sorted_indices[:top_k_output]]

    if len(selected) < top_k_output:
        selected += [""] * (top_k_output - len(selected))

    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Run oracle reranking baseline.")

    parser.add_argument("--input", type=str, required=True, help="Input retrieval results in JSON or JSONL format.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file.")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)

    parser.add_argument(
        "--quantization_mode",
        type=str,
        default="none",
        choices=["none", "bnb4", "mxfp4"],
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="general",
        choices=[
            "general",
            "default",
            "nq",
            "hotpotqa",
            "hotpotnq",
            "2wiki",
            "2wikimultihopqa",
            "musique",
            "medical",
            "covidqa",
            "finance",
            "fever",
            "truthfulqa",
        ],
    )

    parser.add_argument(
        "--truthfulqa_dev_jsonl",
        "--truthful_dev_jsonl",
        dest="truthfulqa_dev_jsonl",
        type=str,
        default=None,
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--top_k_input", type=int, default=10)
    parser.add_argument("--top_k_output", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--append_eos", action="store_true")
    parser.add_argument("--no_append_eos", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_type == "truthfulqa" and not args.truthfulqa_dev_jsonl:
        raise ValueError("--truthfulqa_dev_jsonl is required when --dataset_type truthfulqa.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    generator = LocalCausalLM(
        model_name_or_path=args.model_path,
        tokenizer_name_or_path=args.tokenizer_path,
        quantization_mode=args.quantization_mode,
    )

    generator.tokenizer.padding_side = "right"

    append_eos = bool(args.append_eos) and not bool(args.no_append_eos)

    choice_map = None
    if args.dataset_type == "truthfulqa":
        choice_map = build_truthfulqa_choice_map(args.truthfulqa_dev_jsonl)

    with open(args.output, "w", encoding="utf-8") as fout:
        for item in tqdm(read_json_or_jsonl(args.input), desc="oracle"):
            query_id = extract_query_id(item)
            query = extract_query(item)
            candidates = extract_candidates(item)

            normalized_type = normalize_dataset_type(args.dataset_type)

            if normalized_type == "fever":
                answer = normalize_fever_answer(pick_first_ground_truth(item))
            elif normalized_type == "truthfulqa":
                answer = "_"
            else:
                answer = pick_first_ground_truth(item)

            selected_contexts = oracle_rank_topk(
                generator=generator,
                dataset_type=args.dataset_type,
                query_id=query_id,
                query=query,
                answer=answer,
                candidates=candidates,
                top_k_input=args.top_k_input,
                top_k_output=args.top_k_output,
                batch_size=args.batch_size,
                max_length=args.max_length,
                append_eos=append_eos,
                choice_map=choice_map,
            )

            output_item = {
                "query_id": query_id,
                "query": query,
                "ground_truth": extract_ground_truth(item),
                "selected_contexts": selected_contexts,
                "score_method": "oracle",
                "dataset_type": args.dataset_type,
            }

            write_jsonl_line(fout, output_item)

    print(f"Saved oracle output to: {args.output}")


if __name__ == "__main__":
    main()