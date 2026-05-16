#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train LEAP predictor.

This script trains a dual-stream predictor that combines:
1. Semantic representation from a text encoder.
2. Trace features extracted from LLM internal activations.

The predictor learns to estimate normalized candidate utility scores.
"""

import os
import json
import glob
import math
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LEN = 512
DEFAULT_TOPK = 10
DEFAULT_MAX_STEPS = 128
DEFAULT_TRACE_FEAT_DIM = 7
DEFAULT_EPOCHS = 10
DEFAULT_LR = 2e-5
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-12)


def entropy_from_logits(logits: List[float]) -> float:
    probs = softmax_np(np.array(logits, dtype=np.float64))
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def build_trace_features_for_step(
    step_obj: Dict[str, Any],
    topk: int = DEFAULT_TOPK,
    trace_feat_dim: int = DEFAULT_TRACE_FEAT_DIM,
    fallback_margin: float = 1.0,
) -> Tuple[np.ndarray, int]:
    target_token_id = int(step_obj.get("token_id", -1))
    step_score = float(step_obj.get("step_score", 0.0))
    layers_lens = step_obj.get("layers_lens", [])

    num_layers = len(layers_lens)
    trace_features = np.zeros((num_layers, trace_feat_dim), dtype=np.float32)

    for layer_id in range(num_layers):
        candidate_list = layers_lens[layer_id]
        if not candidate_list:
            continue

        candidate_list = candidate_list[:topk]
        candidate_ids = [int(x.get("token_id", -1)) for x in candidate_list]
        candidate_logits = [float(x.get("logit", 0.0)) for x in candidate_list]

        top1_logit = candidate_logits[0]
        top2_logit = candidate_logits[1] if len(candidate_logits) > 1 else candidate_logits[0]
        top1_top2_gap = top1_logit - top2_logit
        entropy = entropy_from_logits(candidate_logits)

        hit = 1.0 if target_token_id in candidate_ids else 0.0
        if hit:
            target_rank_idx = candidate_ids.index(target_token_id)
            target_logit = candidate_logits[target_rank_idx]
            target_rank = float(target_rank_idx + 1)
        else:
            min_logit = min(candidate_logits)
            target_logit = min_logit - fallback_margin
            target_rank = float(topk + 1)

        margin = top1_logit - target_logit

        trace_features[layer_id, 0] = hit
        trace_features[layer_id, 1] = target_logit
        trace_features[layer_id, 2] = target_rank / float(topk + 1)
        trace_features[layer_id, 3] = margin
        trace_features[layer_id, 4] = top1_top2_gap
        trace_features[layer_id, 5] = entropy
        trace_features[layer_id, 6] = step_score

    return trace_features, num_layers


class CandidateScoreDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        tokenizer,
        max_len: int = DEFAULT_MAX_LEN,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.json")))
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_steps = max_steps

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No JSON files found in: {folder_path}")

        print(f"Found {len(self.file_paths)} files in {folder_path}")

    def __len__(self):
        return len(self.file_paths)

    def _get_text_fields(self, data: Dict[str, Any]) -> Tuple[str, str]:
        question = data.get("question", "")
        if "text" in data:
            context_text = data["text"]
        elif "rag_text" in data:
            context_text = data["rag_text"]
        else:
            context_text = data.get("prompt", "")
        return question, context_text

    def _get_label(self, data: Dict[str, Any]) -> float:
        if "final_score_normal" in data:
            return float(data["final_score_normal"])
        if "final_score" in data:
            return float(data["final_score"])
        return 0.0

    def __getitem__(self, idx):
        for _ in range(3):
            file_path = self.file_paths[idx]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                question, context_text = self._get_text_fields(data)
                label = self._get_label(data)

                encoded = self.tokenizer(
                    question,
                    context_text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt",
                )

                steps = data.get("steps", [])[: self.max_steps]
                query_id = str(data.get("query_id", data.get("id", os.path.basename(file_path))))

                return {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "label": torch.tensor(label, dtype=torch.float32),
                    "steps": steps,
                    "file_path": file_path,
                    "query_id": query_id,
                }

            except Exception:
                idx = (idx + 1) % len(self.file_paths)

        return self.__getitem__((idx + 1) % len(self.file_paths))


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    trace_feats: torch.Tensor
    step_mask: torch.Tensor
    layer_mask: torch.Tensor
    file_paths: List[str]
    query_ids: List[str]


def make_collate_fn(
    topk: int = DEFAULT_TOPK,
    max_steps: int = DEFAULT_MAX_STEPS,
    trace_feat_dim: int = DEFAULT_TRACE_FEAT_DIM,
):
    def collate_fn(batch_list: List[Dict[str, Any]]) -> Batch:
        input_ids = torch.stack([b["input_ids"] for b in batch_list], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch_list], dim=0)
        labels = torch.stack([b["label"] for b in batch_list], dim=0)

        step_lens = [len(b["steps"]) for b in batch_list]
        max_step_len = max(step_lens) if step_lens else 1
        max_step_len = min(max_step_len, max_steps)

        max_layer_len = 0
        for b in batch_list:
            for step in b["steps"]:
                layers_lens = step.get("layers_lens", [])
                if isinstance(layers_lens, list) and len(layers_lens) > 0:
                    max_layer_len = max(max_layer_len, len(layers_lens))
                    break

        if max_layer_len == 0:
            max_layer_len = 1

        trace_feats = torch.zeros(
            (len(batch_list), max_step_len, max_layer_len, trace_feat_dim),
            dtype=torch.float32,
        )
        step_mask = torch.zeros((len(batch_list), max_step_len), dtype=torch.float32)
        layer_mask = torch.zeros((len(batch_list), max_step_len, max_layer_len), dtype=torch.float32)

        for i, b in enumerate(batch_list):
            steps = b["steps"][:max_step_len]
            for t, step in enumerate(steps):
                trace_np, num_layers = build_trace_features_for_step(
                    step,
                    topk=topk,
                    trace_feat_dim=trace_feat_dim,
                )
                num_layers = min(num_layers, max_layer_len)
                trace_feats[i, t, :num_layers, :] = torch.from_numpy(trace_np[:num_layers, :])
                step_mask[i, t] = 1.0
                layer_mask[i, t, :num_layers] = 1.0

        file_paths = [b.get("file_path", "") for b in batch_list]
        query_ids = [b.get("query_id", "") for b in batch_list]

        return Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            trace_feats=trace_feats,
            step_mask=step_mask,
            layer_mask=layer_mask,
            file_paths=file_paths,
            query_ids=query_ids,
        )

    return collate_fn


class LEAPPredictor(nn.Module):
    def __init__(
        self,
        encoder_model_path: str,
        trace_feat_dim: int = DEFAULT_TRACE_FEAT_DIM,
    ):
        super().__init__()

        self.encoder_config = AutoConfig.from_pretrained(encoder_model_path)
        self.encoder_config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(encoder_model_path, config=self.encoder_config)

        self.num_encoder_layers = self.encoder_config.num_hidden_layers + 1
        self.hidden_size = self.encoder_config.hidden_size
        self.trace_feat_dim = trace_feat_dim

        ramp_init = torch.linspace(-5, 5, self.num_encoder_layers)
        self.encoder_layer_weights = nn.Parameter(ramp_init)

        self.semantic_proj = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        self.trace_layer_mlp = nn.Sequential(
            nn.Linear(trace_feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        self.trace_layer_attn = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.trace_token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.trace_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, trace_feats, step_mask, layer_mask):
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = torch.stack(encoder_out.hidden_states, dim=0)
        cls_states = hidden_states[:, :, 0, :]

        layer_weights = torch.softmax(self.encoder_layer_weights, dim=0)
        semantic_embed = torch.sum(cls_states * layer_weights.view(-1, 1, 1), dim=0)
        semantic_feat = self.semantic_proj(semantic_embed)

        batch_size, num_steps, num_layers, feat_dim = trace_feats.shape
        trace_hidden = self.trace_layer_mlp(trace_feats)

        attn_logits = self.trace_layer_attn(trace_hidden).squeeze(-1)
        attn_logits = attn_logits.masked_fill(layer_mask <= 0, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        step_repr = torch.sum(trace_hidden * attn.unsqueeze(-1), dim=2)

        src_key_padding_mask = step_mask <= 0
        step_repr = self.trace_token_encoder(
            step_repr,
            src_key_padding_mask=src_key_padding_mask,
        )

        denom = step_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        trace_embed = (step_repr * step_mask.unsqueeze(-1)).sum(dim=1) / denom
        trace_feat = self.trace_proj(trace_embed)

        fused_feat = torch.cat([semantic_feat, trace_feat], dim=-1)
        score = self.head(fused_feat).squeeze(-1)
        return score


def save_scatter_plot(y_true, y_pred, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=15)

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    plt.title(f"Epoch {epoch}: Ground Truth vs Prediction")
    plt.xlabel("Ground Truth Score")
    plt.ylabel("Predicted Score")
    plt.grid(True, linestyle="--", alpha=0.7)

    file_path = os.path.join(save_dir, f"scatter_epoch_{epoch}.png")
    plt.savefig(file_path)
    plt.close()

    print(f"Scatter plot saved to {file_path}")


@torch.no_grad()
def evaluate(model, loader, device, save_pred_path: Optional[str] = None):
    model.eval()

    preds_all = []
    labels_all = []
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    rows = []

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        trace_feats = batch.trace_feats.to(device)
        step_mask = batch.step_mask.to(device)
        layer_mask = batch.layer_mask.to(device)
        labels = batch.labels.to(device)

        preds = model(input_ids, attention_mask, trace_feats, step_mask, layer_mask)
        loss = loss_fn(preds, labels)
        total_loss += float(loss.item())

        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        preds_all.append(preds_np)
        labels_all.append(labels_np)

        if save_pred_path is not None:
            for file_path, query_id, pred, label in zip(
                batch.file_paths,
                batch.query_ids,
                preds_np.tolist(),
                labels_np.tolist(),
            ):
                rows.append(
                    {
                        "file_path": file_path,
                        "query_id": query_id,
                        "pred": float(pred),
                        "label": float(label),
                    }
                )

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    mse = float(np.mean((preds_all - labels_all) ** 2))
    mae = float(np.mean(np.abs(preds_all - labels_all)))

    if len(np.unique(preds_all)) > 1 and len(np.unique(labels_all)) > 1:
        pearson, _ = pearsonr(preds_all, labels_all)
        spearman, _ = spearmanr(preds_all, labels_all)
    else:
        pearson, spearman = 0.0, 0.0

    if save_pred_path is not None:
        os.makedirs(os.path.dirname(save_pred_path), exist_ok=True)
        with open(save_pred_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_samples": len(rows),
                    "rows": rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved test predictions to: {save_pred_path}")

    return {
        "loss": total_loss / max(1, len(loader)),
        "mse": mse,
        "mae": mae,
        "pearson": float(pearson),
        "spearman": float(spearman),
        "preds_array": preds_all,
        "labels_array": labels_all,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train LEAP predictor.")

    parser.add_argument("--encoder_model_path", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--plot_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--trace_feat_dim", type=int, default=DEFAULT_TRACE_FEAT_DIM)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--test_pred_path",
        type=str,
        default="",
        help="If set, save per-sample test predictions to this JSON file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_path)

    collate_fn = make_collate_fn(
        topk=args.topk,
        max_steps=args.max_steps,
        trace_feat_dim=args.trace_feat_dim,
    )

    train_dataset = CandidateScoreDataset(
        args.train_dir,
        tokenizer,
        max_len=args.max_len,
        max_steps=args.max_steps,
    )
    test_dataset = CandidateScoreDataset(
        args.test_dir,
        tokenizer,
        max_len=args.max_len,
        max_steps=args.max_steps,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = LEAPPredictor(
        encoder_model_path=args.encoder_model_path,
        trace_feat_dim=args.trace_feat_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_spearman = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for batch in train_loop:
            optimizer.zero_grad()

            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            trace_feats = batch.trace_feats.to(device)
            step_mask = batch.step_mask.to(device)
            layer_mask = batch.layer_mask.to(device)
            labels = batch.labels.to(device)

            preds = model(input_ids, attention_mask, trace_feats, step_mask, layer_mask)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_loop.set_postfix(loss=float(loss.item()))

        avg_train_loss = train_loss / max(1, len(train_loader))

        save_pred_path = args.test_pred_path.strip() or None
        metrics = evaluate(model, test_loader, device, save_pred_path=save_pred_path)

        print(f"\n>>> Epoch {epoch} Report")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Test  Loss: {metrics['loss']:.6f}")
        print(f"MSE       : {metrics['mse']:.6f}")
        print(f"MAE       : {metrics['mae']:.6f}")
        print(f"Pearson   : {metrics['pearson']:.4f}")
        print(f"Spearman  : {metrics['spearman']:.4f}")

        save_scatter_plot(
            metrics["labels_array"],
            metrics["preds_array"],
            epoch,
            args.plot_dir,
        )

        if metrics["spearman"] > best_spearman:
            best_spearman = metrics["spearman"]
            torch.save(model.state_dict(), args.save_path)
            print(
                f">>> New best model saved to {args.save_path} "
                f"(best Spearman={best_spearman:.4f})"
            )

        print("-" * 60)


if __name__ == "__main__":
    main()