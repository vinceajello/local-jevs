"""Training and Fine-Tuning Script for Laya System 1 Decision Models.

Implements RLCD (Reinforcement Learning with Calibrated Decisions) using
strictly proper scoring rule rewards (log score + spherical score + ranked
probability score) and soft cross-entropy guidance, followed by post-training
temperature calibration.

Supports:
- Hugging Face datasets (e.g. 'LocalLLaMA/typed-decisions')
- Local JSON/JSONL datasets
- Built-in synthetic demo dataset (--demo)
- Single GPU (CUDA), Apple Silicon (MPS), Multi-GPU DDP (torchrun), and CPU fallback
- Direct export into a checkpoint directory loadable by laya.Router and laya.Agent
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

import laya
from laya.agent import _fix_tokenizer_config, _verify_compatibility
from laya.common import (
    QTYPES,
    QTYPE_NAMES,
    TEMP_MAX,
    TEMP_MIN,
    build_model,
    build_sequence,
    clamp_temperature,
    ece_score,
    proper_reward,
    render_options,
)


def collate_train_batch(items: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Collate preprocessed training items into aligned PyTorch batch tensors."""
    n = len(items)
    max_seq_len = max(len(it["ids"]) for it in items)
    max_markers = max(len(it["markers"]) for it in items)

    input_ids = torch.full((n, max_seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_seq_len), dtype=torch.long)
    marker_pos = torch.zeros((n, max_markers), dtype=torch.long)
    marker_mask = torch.zeros((n, max_markers), dtype=torch.bool)
    target = torch.zeros((n, max_markers), dtype=torch.float32)

    for i, it in enumerate(items):
        seq_len = len(it["ids"])
        input_ids[i, :seq_len] = torch.tensor(it["ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        k = len(it["markers"])
        marker_pos[i, :k] = torch.tensor(it["markers"], dtype=torch.long)
        marker_mask[i, :k] = True

        target_vals = it.get("target", [])
        target[i, : len(target_vals)] = torch.tensor(target_vals, dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items], dtype=torch.long),
        "label": torch.tensor([it.get("label", -1) for it in items], dtype=torch.long),
    }


def build_training_item(
    tok: AutoTokenizer,
    state: Union[str, dict, list],
    q: Dict[str, Any],
    gold_q: Dict[str, Any],
    max_len: int = 512,
    head_max_len: int = 192,
) -> Optional[Dict[str, Any]]:
    """Convert one state + question + gold target tuple into token IDs and markers."""
    t = q.get("type", "choice")
    crit = q.get("criteria", {})
    if t not in QTYPES:
        return None

    if t == "choice":
        keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        probs = gold_q.get("probabilities", {}) if isinstance(gold_q, dict) else {}
        if isinstance(probs, dict) and any(k in probs for k in keys):
            target = [float(probs.get(k, 0.0)) for k in keys]
        elif isinstance(gold_q, dict) and "choice" in gold_q:
            c = gold_q["choice"]
            target = [1.0 if k == c else 0.0 for k in keys]
        elif isinstance(gold_q, (str, int)):
            target = [1.0 if str(k) == str(gold_q) else 0.0 for k in keys]
        else:
            target = [1.0 / max(1, len(keys))] * len(keys)

    elif t == "noul":
        probs = gold_q.get("probabilities", {}) if isinstance(gold_q, dict) else {}
        if isinstance(probs, dict) and ("false" in probs or "true" in probs):
            target = [float(probs.get("false", 0.5)), float(probs.get("true", 0.5))]
        elif isinstance(gold_q, dict) and "noul" in gold_q:
            val = float(gold_q["noul"])
            target = [1.0 - val, val]
        elif isinstance(gold_q, bool):
            target = [0.0, 1.0] if gold_q else [1.0, 0.0]
        else:
            target = [0.5, 0.5]

    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 5
        probs = gold_q.get("probabilities", {}) if isinstance(gold_q, dict) else {}
        if isinstance(probs, dict) and any(str(i) in probs for i in range(n_levels)):
            target = [float(probs.get(str(i), 0.0)) for i in range(n_levels)]
        elif isinstance(gold_q, dict) and "score" in gold_q:
            s_val = int(round(float(gold_q["score"])))
            target = [1.0 if i == s_val else 0.0 for i in range(n_levels)]
        elif isinstance(gold_q, (int, float)):
            s_val = int(round(float(gold_q)))
            target = [1.0 if i == s_val else 0.0 for i in range(n_levels)]
        else:
            target = [1.0 / n_levels] * n_levels

    s = sum(target)
    if s > 0:
        target = [v / s for v in target]
    else:
        target = [1.0 / len(target)] * len(target)

    label = int(np.argmax(target))
    opts_rendered = render_options({"t": t, "crit": crit})
    k = len(opts_rendered)

    seq, markers = build_sequence(
        tok,
        state,
        {"t": t, "ins": q.get("instructions", ""), "crit": crit},
        max_len=max_len,
        head_max_len=head_max_len,
    )
    if len(markers) != k:
        return None

    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[t],
        "target": target,
        "label": label,
    }


def generate_synthetic_demo_data() -> List[Dict[str, Any]]:
    """Generate realistic demonstration cases across choice, score, and noul tasks."""
    demo_cases = [
        {
            "state": {
                "from": "alice@partner.org",
                "subject": "System downtime: payment gateway returning 500 error",
                "body": "Our checkout is failing for all European customers. Emergency escalation needed immediately!",
            },
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this incident?",
                    "criteria": {
                        "technical": "outages, server errors, bugs",
                        "billing": "invoices, subscriptions, refunds",
                        "sales": "new deals, partnership requests",
                        "general": "other requests",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate incident urgency.",
                    "criteria": ["low", "normal", "high", "critical blocker"],
                },
                "is_escalation": {
                    "type": "noul",
                    "instructions": "Does this require emergency escalation?",
                },
            },
            "gold": {
                "department": {"choice": "technical", "probabilities": {"technical": 0.95, "billing": 0.03, "sales": 0.01, "general": 0.01}},
                "urgency": {"score": 3, "probabilities": {"0": 0.01, "1": 0.04, "2": 0.15, "3": 0.80}},
                "is_escalation": {"noul": 0.96, "probabilities": {"false": 0.04, "true": 0.96}},
            },
        },
        {
            "state": {
                "from": "finance@client.com",
                "subject": "Invoice dispute #8820",
                "body": "We were charged twice for monthly licenses. Kindly issue a credit memo or refund.",
            },
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this incident?",
                    "criteria": {
                        "technical": "outages, server errors, bugs",
                        "billing": "invoices, subscriptions, refunds",
                        "sales": "new deals, partnership requests",
                        "general": "other requests",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate incident urgency.",
                    "criteria": ["low", "normal", "high", "critical blocker"],
                },
                "refund_requested": {
                    "type": "noul",
                    "instructions": "Did the customer request a refund or credit?",
                },
            },
            "gold": {
                "department": {"choice": "billing", "probabilities": {"technical": 0.02, "billing": 0.94, "sales": 0.02, "general": 0.02}},
                "urgency": {"score": 1, "probabilities": {"0": 0.15, "1": 0.70, "2": 0.12, "3": 0.03}},
                "refund_requested": {"noul": 0.98, "probabilities": {"false": 0.02, "true": 0.98}},
            },
        },
        {
            "state": {
                "from": "procurement@enterprise.com",
                "subject": "Enterprise tier quote for 500 seats",
                "body": "We are expanding our team and would like volume pricing details for next fiscal quarter.",
            },
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this incident?",
                    "criteria": {
                        "technical": "outages, server errors, bugs",
                        "billing": "invoices, subscriptions, refunds",
                        "sales": "new deals, partnership requests",
                        "general": "other requests",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate incident urgency.",
                    "criteria": ["low", "normal", "high", "critical blocker"],
                },
                "churn_risk": {
                    "type": "noul",
                    "instructions": "Is the customer threatening to churn or cancel?",
                },
            },
            "gold": {
                "department": {"choice": "sales", "probabilities": {"technical": 0.01, "billing": 0.04, "sales": 0.93, "general": 0.02}},
                "urgency": {"score": 1, "probabilities": {"0": 0.20, "1": 0.65, "2": 0.12, "3": 0.03}},
                "churn_risk": {"noul": 0.03, "probabilities": {"false": 0.97, "true": 0.03}},
            },
        },
        {
            "state": {
                "from": "user@retail.com",
                "subject": "Feature request: dark mode support",
                "body": "Love the application! Would it be possible to add a dark theme option in settings?",
            },
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this incident?",
                    "criteria": {
                        "technical": "outages, server errors, bugs",
                        "billing": "invoices, subscriptions, refunds",
                        "sales": "new deals, partnership requests",
                        "general": "other requests",
                    },
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate incident urgency.",
                    "criteria": ["low", "normal", "high", "critical blocker"],
                },
                "churn_risk": {
                    "type": "noul",
                    "instructions": "Is the customer threatening to churn or cancel?",
                },
            },
            "gold": {
                "department": {"choice": "general", "probabilities": {"technical": 0.15, "billing": 0.01, "sales": 0.02, "general": 0.82}},
                "urgency": {"score": 0, "probabilities": {"0": 0.85, "1": 0.12, "2": 0.02, "3": 0.01}},
                "churn_risk": {"noul": 0.01, "probabilities": {"false": 0.99, "true": 0.01}},
            },
        },
    ]
    # Multiply demo cases to form a full test dataset
    expanded = []
    for _ in range(15):
        expanded.extend(copy.deepcopy(demo_cases))
    return expanded


def load_dataset_items(
    args: argparse.Namespace, tok: AutoTokenizer, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Load items from Hugging Face dataset, local file, or synthetic demo generator."""
    raw_cases: List[Dict[str, Any]] = []

    if args.demo:
        print("[train.py] Using built-in synthetic demo dataset...")
        raw_cases = generate_synthetic_demo_data()

    elif args.dataset_path:
        print(f"[train.py] Loading dataset from local file: {args.dataset_path}...")
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Local dataset path not found: {args.dataset_path}")
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            if args.dataset_path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        raw_cases.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    raw_cases = data
                else:
                    raw_cases = [data]

    elif args.dataset_name:
        print(f"[train.py] Loading Hugging Face dataset '{args.dataset_name}' (split: {args.dataset_split})...")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required to load datasets from Hugging Face. "
                "Install it using: uv add datasets or pip install datasets"
            )
        hf_ds = load_dataset(args.dataset_name, "all", split=args.dataset_split)
        for row in hf_ds:
            state = json.loads(row["state"]) if isinstance(row.get("state"), str) else row.get("state")
            questions = json.loads(row["questions"]) if isinstance(row.get("questions"), str) else row.get("questions")
            gold = json.loads(row["gold"]) if isinstance(row.get("gold"), str) else row.get("gold")
            raw_cases.append({"state": state, "questions": questions, "gold": gold})
    else:
        print("[train.py] No dataset specified. Defaulting to synthetic demo data (--demo).")
        raw_cases = generate_synthetic_demo_data()

    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)
    items: List[Dict[str, Any]] = []

    for case in raw_cases:
        state = case.get("state", "")
        questions = case.get("questions", {})
        gold = case.get("gold", case.get("answers", {}))
        for qid, q in questions.items():
            if qid in gold:
                it = build_training_item(tok, state, q, gold[qid], max_len, head_max_len)
                if it:
                    items.append(it)

    if args.max_samples and len(items) > args.max_samples:
        print(f"[train.py] Limiting dataset to max_samples={args.max_samples} (from {len(items)} items).")
        items = items[: args.max_samples]

    print(f"[train.py] Successfully preprocessed {len(items)} training sequences.")
    return items


def fit_one_temp(sel: List[Tuple[np.ndarray, List[float]]]) -> float:
    """Fit optimal temperature using L-BFGS to minimize NLL on calibrated items."""
    if len(sel) < 5:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    z_tensor = torch.full((len(sel), kmax), -1e4, dtype=torch.float32)
    t_tensor = torch.zeros((len(sel), kmax), dtype=torch.float32)

    for i, (z, t) in enumerate(sel):
        z_tensor[i, : len(z)] = torch.tensor(z, dtype=torch.float32)
        t_tensor[i, : len(t)] = torch.tensor(t, dtype=torch.float32)

    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temp = log_t.exp()
        scaled_logits = z_tensor / temp
        log_probs = torch.log_softmax(scaled_logits, dim=-1)
        loss = -(t_tensor * log_probs).sum(dim=-1).mean()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        t_val = float(torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item())
        return clamp_temperature(t_val)
    except Exception:
        return 1.0


def resolve_device(requested_device: Optional[str] = None) -> torch.device:
    """Resolve target training device with safe hardware fallback."""
    if requested_device and requested_device != "auto":
        target = torch.device(requested_device)
        if target.type == "cuda" and not torch.cuda.is_available():
            print("[train.py] Warning: CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        if target.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("[train.py] Warning: MPS requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return target

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> str:
    """Execute complete training, checkpointing, and calibration loop."""
    # 1. Distributed setup detection
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        torch.distributed.init_process_group(backend=backend)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        device = resolve_device(args.device)

    is_main_process = rank == 0

    if is_main_process:
        print(f"=== Starting Laya Training ===")
        print(f"Device: {device} | Distributed: {is_distributed} (rank {rank})")
        print(f"Base Model: {args.model_id} | Output: {args.output_dir}")

    # 2. Download and load base checkpoint
    model_dir = args.model_id
    if not os.path.exists(model_dir):
        if is_main_process:
            print(f"[train.py] Fetching base model checkpoint from Hugging Face: {args.model_id}...")
        subfolder_prefix = f"{args.subfolder}/" if args.subfolder else ""
        patterns = [
            subfolder_prefix + p
            for p in ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
        ]
        model_dir = snapshot_download(
            args.model_id,
            allow_patterns=patterns,
            token=args.hf_token or os.environ.get("HF_TOKEN"),
        )
    if args.subfolder:
        model_dir = os.path.join(model_dir, args.subfolder)

    _fix_tokenizer_config(model_dir)

    cfg_file = os.path.join(model_dir, "rl_agent_config.json")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"rl_agent_config.json not found in {model_dir}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tok_path = os.path.join(model_dir, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_path if os.path.exists(tok_path) else cfg.get("encoder"))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    enc_path = os.path.join(model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_path if os.path.exists(enc_path) else None)

    weights_file = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(weights_file):
        weights = load_file(weights_file)
        _verify_compatibility(model, cfg, weights, args.model_id)
        model.load_state_dict(weights, strict=True)
    else:
        if is_main_process:
            print("[train.py] Warning: model.safetensors not found; initializing from base encoder weights.")

    if hasattr(model.encoder.config, "reference_compile"):
        model.encoder.config.reference_compile = False

    model.to(device)

    # 3. Load Dataset
    items = load_dataset_items(args, tok, cfg)
    if not items:
        raise ValueError("No valid training items could be built from dataset.")

    # Split off validation/calibration split (last 10% or up to 200 items)
    split_idx = max(1, int(len(items) * 0.9))
    train_items = items[:split_idx]
    calib_items = items[split_idx:] if split_idx < len(items) else items[:min(50, len(items))]

    if is_distributed:
        world_size = torch.distributed.get_world_size()
        train_items = train_items[rank::world_size]

    # Wrap model with DDP if distributed
    training_model = model
    if is_distributed:
        training_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    # 4. Optimizer, Scheduler, and Scaler
    enc_params = [p for n, p in training_model.named_parameters() if "encoder." in n and p.requires_grad]
    head_params = [p for n, p in training_model.named_parameters() if "encoder." not in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )

    total_steps = (len(train_items) // (args.batch_size * args.grad_accum) + 1) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps), eta_min=1e-6
    )

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5. Training Loop (RLCD + CE Guidance)
    t0_start = time.time()
    micro_batch = args.batch_size
    grad_accum = args.grad_accum
    group_size = args.group_size

    for epoch in range(args.epochs):
        training_model.train()
        random.seed(42 + epoch + rank)
        random.shuffle(train_items)

        epoch_loss = 0.0
        n_batches = 0
        accum_step = 0
        optimizer.zero_grad(set_to_none=True)

        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress

        for b_idx in range(0, len(train_items), micro_batch):
            chunk = train_items[b_idx : b_idx + micro_batch]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            target = batch["target"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, act = training_model(
                    input_ids, attention_mask, marker_pos, marker_mask, qtype
                )

            logits = logits.float()
            k = marker_mask.sum(-1, keepdim=True).float().clamp(min=1.0)

            # RLCD: Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * marker_mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * marker_mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~marker_mask.unsqueeze(0), -1e4), -1)

            # Proper scoring rule reward (w_sph=0.75 for soft distribution matching)
            with torch.no_grad():
                r = proper_reward(
                    q, target.unsqueeze(0), qtype, marker_mask, w_sph=0.75, w_rps=1.0
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # Policy gradient loss + Soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * marker_mask.unsqueeze(0)).sum(-1) / (
                2 * (sigma**2)
            )
            loss_rl = -(adv * logp).mean()

            log_probs = torch.log_softmax(logits.masked_fill(~marker_mask, -1e4), -1)
            loss_ce = -(target * log_probs).sum(-1).mean()

            loss = (loss_rl + args.ce_weight * loss_ce) / grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_step += 1
            if accum_step % grad_accum == 0 or (b_idx + micro_batch) >= len(train_items):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(training_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(training_model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum
            n_batches += 1

            if is_main_process and (n_batches % max(1, args.log_interval)) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | Step {n_batches} | "
                    f"Loss: {loss.item()*grad_accum:.4f} (RL: {loss_rl.item():.4f}, CE: {loss_ce.item():.4f}) | "
                    f"Reward: {r.mean().item():.3f} | LR: {cur_lr:.2e}"
                )

        if is_main_process:
            avg_loss = epoch_loss / max(1, n_batches)
            print(
                f"=== Epoch {epoch+1}/{args.epochs} Completed in {time.time()-t0_start:.1f}s | Avg Loss: {avg_loss:.4f} ==="
            )

            # Save rolling checkpoint
            ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_model_artifacts(model, tok, cfg, ckpt_dir)

        if is_distributed:
            torch.distributed.barrier()

    # 6. Post-training Temperature Calibration (Rank 0)
    fitted_temps = [1.0, 1.0, 1.0]
    if is_main_process:
        print("\nFitting post-training calibration temperatures on holdout split...")
        model.eval()
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), micro_batch):
                c_chunk = calib_items[c_idx : c_idx + micro_batch]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                l_np = l_sub.float().cpu().numpy()
                for r_idx, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r_idx, :k], it["target"]))

        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print(
                f"Fitted calibration temperatures [choice, score, noul]: "
                f"{[round(t, 4) for t in fitted_temps]}"
            )
        except Exception as e:
            print(f"Calibration fitting note: {e}")

        # Final export
        cfg["fine_tuned"] = True
        cfg["temperature"] = fitted_temps
        save_model_artifacts(model, tok, cfg, args.output_dir)
        print(f"\nTraining complete! Successfully saved model to {args.output_dir}")

    if is_distributed:
        torch.distributed.destroy_process_group()

    return args.output_dir


def save_model_artifacts(
    model: nn.Module, tok: AutoTokenizer, cfg: Dict[str, Any], output_dir: str
):
    """Save all weights, configs, tokenizer, and encoder to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save safetensors
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(output_dir, "model.safetensors"))

    # Save tokenizer
    tok_dir = os.path.join(output_dir, "tokenizer")
    tok.save_pretrained(tok_dir)

    # Save encoder config
    enc_dir = os.path.join(output_dir, "encoder")
    os.makedirs(enc_dir, exist_ok=True)
    model.encoder.config.save_pretrained(enc_dir)

    # Save Laya RL Agent config
    clean_cfg = copy.deepcopy(cfg)
    if "temperature_by_options" in clean_cfg and isinstance(clean_cfg["temperature_by_options"], dict):
        clean_cfg["temperature_by_options"] = {
            k: clamp_temperature(v) for k, v in clean_cfg["temperature_by_options"].items()
        }
    with open(os.path.join(output_dir, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        json.dump(clean_cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Train and fine-tune Laya System 1 decision models with RLCD."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="convaiinnovations/laya",
        help="Base Laya model ID on Hugging Face or local path.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder in the repository (e.g. 'multilingual', 'typed-decisions').",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name (e.g. 'LocalLLaMA/typed-decisions').",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to train on (default: 'train').",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to local JSON or JSONL dataset file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Train on built-in synthetic demonstration cases.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./laya_finetuned",
        help="Output directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs (default: 4).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Micro-batch size per forward pass (default: 8).",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--lr_encoder",
        type=float,
        default=2.5e-5,
        help="Encoder learning rate (default: 2.5e-5).",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=1.0e-4,
        help="Decision head learning rate (default: 1.0e-4).",
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="Weight for soft cross-entropy guidance loss (default: 1.0).",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="GRPO group size for exploration noise sampling (default: 4).",
    )
    parser.add_argument(
        "--sigma_start",
        type=float,
        default=0.4,
        help="Initial exploration noise sigma (default: 0.4).",
    )
    parser.add_argument(
        "--sigma_end",
        type=float,
        default=0.1,
        help="Final exploration noise sigma (default: 0.1).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum training items to use (for fast testing).",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Steps between training log outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face access token.",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
