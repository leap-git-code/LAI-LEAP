#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility wrapper for local causal language models used by LLM-based rerankers.
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig

try:
    from transformers import Mxfp4Config
except ImportError:
    Mxfp4Config = None


class LocalCausalLM:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        quantization_mode: str = "none",
    ):
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        try:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model_type = (getattr(config, "model_type", "") or "").lower()
        except Exception:
            model_type = ""

        model_path_lower = model_name_or_path.lower()

        if "qwen" in model_type or "qwen" in model_path_lower:
            self.model_family = "qwen"
        elif "llama" in model_type or "llama" in model_path_lower:
            self.model_family = "llama"
        elif "gpt" in model_type or "gpt" in model_path_lower or "oss" in model_path_lower:
            self.model_family = "gpt"
        else:
            self.model_family = "auto"

        if quantization_mode == "mxfp4":
            if Mxfp4Config is None:
                raise ImportError(
                    "Mxfp4Config is not available. Please upgrade transformers "
                    "or use quantization_mode='none' or 'bnb4'."
                )
            quant_config = Mxfp4Config(dequantize=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
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
                model_name_or_path,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
            ).eval()

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()

        self.harmony_final_prefix = "<|channel|>final<|message|>"

    def apply_chat_template(self, system_prompt: str, user_prompt: str) -> str:
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
        for token in ["<|im_start|>", "<|start_header_id|>"]:
            token_id = self._safe_token_id(token)
            if token_id is not None:
                bad_words.append([token_id])
        return bad_words if bad_words else None

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
        ).to(self.model.device)

        do_sample = temperature > 0.0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.get_stop_ids() or None,
            pad_token_id=self.tokenizer.pad_token_id,
            bad_words_ids=self.get_bad_words_ids(),
            use_cache=True,
        )

        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0, input_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()