#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GlobalEncoder(nn.Module):
    """Global interaction encoder over one query and its candidate contexts."""

    def __init__(
        self,
        num_layers: int,
        global_hidden_size: int,
        num_heads: int,
        num_labels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=global_hidden_size,
            nhead=num_heads,
            batch_first=True,
            activation=F.gelu,
            norm_first=False,
            dropout=dropout,
        )

        self.context_attn = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.query_embedding = nn.Embedding(2, global_hidden_size)
        self.query_norm = nn.LayerNorm(global_hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.score_head = nn.Sequential(
            nn.Linear(global_hidden_size * 2, global_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_size, num_labels),
        )

    def forward(
        self,
        pair_feat: torch.Tensor,
        pair_nums: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(set(pair_nums)) == 1:
            group_size = pair_nums[0]
            num_groups = len(pair_nums)
            batch_pair_feat = pair_feat.view(num_groups, group_size, -1)
        else:
            batch_pair_feat = nn.utils.rnn.pad_sequence(
                pair_feat.split(pair_nums),
                batch_first=True,
            )

        batch_size, max_len = batch_pair_feat.shape[:2]

        query_tags = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        query_tags[:, 0] = 1
        batch_pair_feat = self.query_norm(
            batch_pair_feat + self.query_embedding(query_tags)
        )

        pair_list_feat = self.context_attn(
            batch_pair_feat,
            src_key_padding_mask=self.attention_mask(pair_nums).to(device),
            mask=self.query_isolated_attention_mask(pair_nums).to(device),
        )

        query_feat = pair_list_feat[:, 0, :]

        if len(set(pair_nums)) == 1:
            group_size = pair_nums[0]
            context_feat = pair_list_feat[:, 1:group_size, :]
            query_expanded = query_feat.unsqueeze(1).expand(
                -1,
                group_size - 1,
                -1,
            )
            pair_feat_cat = torch.cat([query_expanded, context_feat], dim=-1)
            pair_feat_cat = pair_feat_cat.reshape(-1, pair_feat_cat.size(-1))
        else:
            pair_feat_list = [
                pair_list_feat[index, 1:num_items, :]
                for index, num_items in enumerate(pair_nums)
            ]
            pair_feat_cat = torch.cat(
                [
                    torch.cat(
                        [
                            query_feat[index].unsqueeze(0).repeat(num_items - 1, 1),
                            pair_feat_list[index],
                        ],
                        dim=1,
                    )
                    for index, num_items in enumerate(pair_nums)
                ],
                dim=0,
            )

        pair_feat_cat = self.dropout(pair_feat_cat)
        logits = self.score_head(pair_feat_cat)

        valid_hidden = torch.cat(
            [
                pair_list_feat[index, :num_items, :]
                for index, num_items in enumerate(pair_nums)
            ],
            dim=0,
        )

        return logits, valid_hidden

    @staticmethod
    def attention_mask(pair_nums: List[int]) -> torch.Tensor:
        max_len = max(pair_nums)
        lengths = torch.tensor(pair_nums)
        mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask

    @staticmethod
    def query_isolated_attention_mask(pair_nums: List[int]) -> torch.Tensor:
        max_len = max(pair_nums)
        mask = torch.zeros(max_len, max_len, dtype=torch.bool)
        mask[0, 1:] = True
        return mask


class GroupBatchSampler(BatchSampler):
    """Batch sampler that keeps each query-context group intact."""

    def __init__(
        self,
        dataset_len: int,
        group_size: int = 11,
        groups_per_batch: int = 1,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        if dataset_len % group_size != 0:
            raise ValueError(
                f"dataset_len must be divisible by group_size, got "
                f"dataset_len={dataset_len}, group_size={group_size}."
            )

        self.dataset_len = dataset_len
        self.group_size = group_size
        self.num_groups = dataset_len // group_size
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        group_indices = list(range(self.num_groups))

        if self.shuffle:
            generator = self.generator or torch.Generator()
            perm = torch.randperm(self.num_groups, generator=generator).tolist()
            group_indices = [group_indices[index] for index in perm]

        for start in range(0, self.num_groups, self.groups_per_batch):
            batch_groups = group_indices[start:start + self.groups_per_batch]
            yield [
                group_index * self.group_size + offset
                for group_index in batch_groups
                for offset in range(self.group_size)
            ]

    def __len__(self) -> int:
        return math.ceil(self.num_groups / self.groups_per_batch)


class CSM(nn.Module):
    """CSM supervised reranker."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__()

        local_hidden_size = int(config["refiner_local_hidden_size"])
        global_hidden_size = int(config["refiner_global_hidden_size"])
        num_heads = int(config["refiner_num_heads"])
        global_layers = int(config["refiner_global_layers"])
        bert_path = config["model2path"]["bert"]

        self.num_labels = 1
        self.context_size = int(config["retrieval_topk"])
        self.group_size = self.context_size + 1
        self.bert_path = bert_path
        self.tokenizer = None

        self.local_encoder = AutoModel.from_pretrained(self.bert_path)
        self.feat_map = nn.Linear(local_hidden_size, global_hidden_size)

        self.global_encoder = GlobalEncoder(
            num_layers=global_layers,
            global_hidden_size=global_hidden_size,
            num_heads=num_heads,
            num_labels=self.num_labels,
        )

        self.proj_head = nn.Sequential(
            nn.Linear(global_hidden_size, global_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(global_hidden_size // 2, 64),
        )

        self.init_weights()

    def init_weights(self) -> None:
        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        for block in [self.feat_map, self.global_encoder, self.proj_head]:
            block.apply(_init)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[SequenceClassifierOutput, torch.Tensor]:
        device = input_ids.device

        pooled = kwargs.pop("pooled", None)

        if pooled is None:
            encoder_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }

            if token_type_ids is not None:
                encoder_inputs["token_type_ids"] = token_type_ids

            outputs = self.local_encoder(**encoder_inputs)
            pooled = self.avg_pooling(outputs.last_hidden_state, attention_mask)

        pair_feat = self.feat_map(pooled)

        batch_size = len(input_ids)
        num_groups = batch_size // self.group_size
        usable_size = num_groups * self.group_size

        pair_feat = pair_feat[:usable_size]
        logits, hidden_repr = self.global_encoder(
            pair_feat,
            [self.group_size] * num_groups,
            device,
        )

        context_repr = torch.cat(
            [
                hidden_repr[
                    group_index * self.group_size + 1:
                    (group_index + 1) * self.group_size
                ]
                for group_index in range(num_groups)
            ],
            dim=0,
        )

        context_repr = F.normalize(self.proj_head(context_repr), p=2, dim=1)
        context_repr = context_repr.view(num_groups, self.context_size, -1)

        if self.num_labels == 1:
            logits = torch.tanh(logits)

        return SequenceClassifierOutput(loss=None, logits=logits), context_repr

    @staticmethod
    def avg_pooling(
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).to(hidden_state.dtype)
        sum_embeddings = torch.sum(hidden_state * mask, dim=1)
        sum_mask = mask.sum(dim=1).clamp_min(1e-12)
        return sum_embeddings / sum_mask

    @staticmethod
    def get_group_sampler(
        dataset_len: int,
        batch_size: int,
        group_size: int = 11,
        shuffle: bool = True,
    ) -> GroupBatchSampler:
        groups_per_batch = max(1, batch_size // group_size)
        return GroupBatchSampler(
            dataset_len=dataset_len,
            group_size=group_size,
            groups_per_batch=groups_per_batch,
            shuffle=shuffle,
        )

    def load_model(self, model_dir: str) -> None:
        checkpoint_path = os.path.join(model_dir, "model.safetensors")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")

        state_dict = load_file(checkpoint_path)
        self.load_state_dict(state_dict)

    def fit_st(
        self,
        num_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-4,
        cts_loss_w: float = 1.0,
        temp: float = 0.1,
        save_dir: Optional[str] = None,
        num_workers: int = 4,
    ) -> None:
        if save_dir is None:
            raise ValueError("save_dir must be specified for CSM training.")

        train_data_path = os.path.join(save_dir, "train_ds.parquet")
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training parquet file not found: {train_data_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path, use_fast=True)
        group_size = self.group_size
        context_size = self.context_size

        for parameter in self.local_encoder.parameters():
            parameter.requires_grad = False

        train_dataset = Dataset.from_parquet(train_data_path)

        train_args = TrainingArguments(
            output_dir=save_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size * group_size,
            per_device_eval_batch_size=batch_size * group_size,
            num_train_epochs=num_epochs,
            logging_steps=200,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            dataloader_drop_last=True,
            lr_scheduler_type="cosine",
            warmup_ratio=0.01,
            weight_decay=0.005,
            max_grad_norm=1.5,
            report_to=[],
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )

        model_ref = self

        class CSMTrainer(Trainer):
            def get_train_dataloader(self):
                sampler = CSM.get_group_sampler(
                    dataset_len=len(self.train_dataset),
                    batch_size=self.args.per_device_train_batch_size,
                    group_size=group_size,
                    shuffle=True,
                )

                return DataLoader(
                    self.train_dataset,
                    batch_sampler=sampler,
                    collate_fn=self.data_collator,
                    num_workers=num_workers,
                    pin_memory=True,
                )

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")

                outputs, context_repr = model(**inputs)
                logits = outputs.logits.view(-1)

                target = labels.view(-1, group_size)[:, 1:]
                target = target.reshape(-1).to(logits.dtype)
                target = target[:len(logits)]

                regression_loss = F.smooth_l1_loss(
                    logits,
                    target,
                    beta=0.1,
                )

                num_groups = len(logits) // context_size
                if num_groups > 0:
                    contrastive_loss = self.compute_context_loss(
                        context_repr=context_repr,
                        targets=target[:num_groups * context_size].view(
                            num_groups,
                            context_size,
                        ),
                    )
                else:
                    contrastive_loss = torch.tensor(0.0, device=logits.device)

                loss = regression_loss + cts_loss_w * contrastive_loss
                return (loss, outputs) if return_outputs else loss

            def compute_context_loss(
                self,
                context_repr: torch.Tensor,
                targets: torch.Tensor,
            ) -> torch.Tensor:
                device = context_repr.device

                signs = torch.sign(targets)
                positive_mask = signs > 1e-4
                negative_mask = signs < -1e-4

                valid_groups = positive_mask.any(dim=1) & negative_mask.any(dim=1)
                if not valid_groups.any():
                    return torch.tensor(0.0, device=device)

                context_repr = context_repr[valid_groups]
                positive_mask = positive_mask[valid_groups]
                negative_mask = negative_mask[valid_groups]

                first_positive_index = torch.argmax(positive_mask.long(), dim=1)
                anchor = context_repr[
                    torch.arange(context_repr.size(0), device=device),
                    first_positive_index,
                ]

                similarity = torch.bmm(
                    anchor.unsqueeze(1),
                    context_repr.transpose(1, 2),
                ).squeeze(1) / temp

                valid_mask = positive_mask | negative_mask

                similarity_all = similarity.clone()
                similarity_all[~valid_mask] = float("-inf")
                log_sum_all = torch.logsumexp(similarity_all, dim=1)

                similarity_pos = similarity.clone()
                similarity_pos[~positive_mask] = float("-inf")
                log_sum_pos = torch.logsumexp(similarity_pos, dim=1)

                return (-log_sum_pos + log_sum_all).mean()

        trainer = CSMTrainer(
            model=model_ref,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
            ),
        )

        trainer.train()

    @torch.no_grad()
    def predict(
        self,
        questions: List[str],
        context_lists: List[List[str]],
        batch_size: int = 32,
        device: torch.device = torch.device("cuda"),
        encode_batch_size: int = 256,
        max_length: int = 512,
    ) -> List[List[float]]:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path, use_fast=True)

        self.eval()
        self.to(device)

        all_scores: List[List[float]] = []

        for start in tqdm(range(0, len(questions), batch_size), desc="CSM predicting"):
            batch_questions = questions[start:start + batch_size]
            batch_context_lists = context_lists[start:start + batch_size]

            flat_texts = [
                text
                for question, contexts in zip(batch_questions, batch_context_lists)
                for text in [question] + contexts
            ]

            pair_nums = [len(contexts) + 1 for contexts in batch_context_lists]

            pooled_list = []

            for encode_start in range(0, len(flat_texts), encode_batch_size):
                batch_texts = flat_texts[encode_start:encode_start + encode_batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                encoder_inputs = {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "return_dict": True,
                }

                if "token_type_ids" in encoded:
                    encoder_inputs["token_type_ids"] = encoded["token_type_ids"]

                outputs = self.local_encoder(**encoder_inputs)
                pooled = self.avg_pooling(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                )
                pooled_list.append(pooled)

            pair_feat = self.feat_map(torch.cat(pooled_list, dim=0))
            logits, _ = self.global_encoder(pair_feat, pair_nums, device)

            if self.num_labels == 1:
                logits = torch.tanh(logits)

            flat_scores = logits.squeeze(-1).detach().cpu().tolist()
            if isinstance(flat_scores, float):
                flat_scores = [flat_scores]

            current_index = 0
            for num_items in pair_nums:
                num_contexts = num_items - 1
                all_scores.append(flat_scores[current_index:current_index + num_contexts])
                current_index += num_contexts

        return all_scores
