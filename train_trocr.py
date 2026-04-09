"""
train_trocr.py
==============
TrOCR-based Handwritten Text Recognition on the IAM Handwriting Database.

Architecture : microsoft/trocr-small-handwritten (ViT encoder + GPT-2 decoder)
Hardware     : NVIDIA RTX 3050 4 GB  →  fp16 + gradient accumulation
Dataset      : IAM words  (images in iam_words/words/, labels in iam_words/words.txt)
Metric       : Character Error Rate (CER) via jiwer

Usage
-----
    python train_trocr.py

All paths are configured in the CONFIG dict below.
Checkpoints are saved locally and pushed to GitHub after every epoch.
"""

import os
import sys
import json
import math
import time
import random
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE = None

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)
from jiwer import cer as compute_cer

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── paths ────────────────────────────────────────────────────────────────
    "dataset_root": r"c:\Users\anish\Soft Computing\handwriting dataset\iam_words",
    "words_txt": r"c:\Users\anish\Soft Computing\handwriting dataset\iam_words\words.txt",
    "repo_root": r"c:\Users\anish\Soft Computing\Hand-Written-Text-reg",
    "checkpoint_dir": r"c:\Users\anish\Soft Computing\Hand-Written-Text-reg\checkpoints",
    "log_file": r"c:\Users\anish\Soft Computing\Hand-Written-Text-reg\training_log.json",

    # ── model ────────────────────────────────────────────────────────────────
    "model_name": "microsoft/trocr-small-handwritten",
    "image_size": 384,

    # ── training ─────────────────────────────────────────────────────────────
    "epochs": 20,
    "batch_size": 2,           # per-GPU batch (4 GB VRAM)
    "accum_steps": 8,          # effective batch = 2 × 8 = 16
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.05,      # 5 % of total steps used for warmup
    "fp16": True,

    # ── dataset ──────────────────────────────────────────────────────────────
    "val_split": 0.10,         # 10 % validation, held-out by form ID
    "max_label_len": 32,       # truncate very long tokens
    "num_workers": 0,          # set >0 only if you have fast NVMe

    # ── early stopping ───────────────────────────────────────────────────────
    "patience": 5,

    # ── git ──────────────────────────────────────────────────────────────────
    "git_push": True,
    "git_remote": "origin",
    "git_branch": "main",
}

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(CONFIG["repo_root"]) / "train.log", mode="a"
        ),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
def parse_words_txt(words_txt: str) -> List[Tuple[str, str]]:
    """
    Parse IAM words.txt → list of (image_path, label).

    Line format (after header):
        a01-000u-00-00 ok 154 408 768 27 51 AT A
    Fields: word_id  seg_ok  gray  x  y  w  h  tag  transcription
    We keep only 'ok' segmentation entries.
    """
    records: List[Tuple[str, str]] = []
    dataset_root = Path(CONFIG["dataset_root"])
    words_dir = dataset_root / "words"

    with open(words_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            word_id = parts[0]        # e.g. a01-000u-00-00
            seg = parts[1]            # ok / err
            # transcription = last field (may contain spaces in IAM edge-cases,
            # but standard word-level entries have exactly one word)
            transcription = parts[-1]

            if seg != "ok":
                continue
            if not transcription or transcription in ("#", "&GA"):
                continue

            # build image path: words/{folder}/{subfolder}/{word_id}.png
            # folder   = first part   a01
            # subfolder = first two parts  a01-000u
            split_id = word_id.split("-")
            if len(split_id) < 2:
                continue
            folder = split_id[0]
            subfolder = "-".join(split_id[:2])
            img_path = words_dir / folder / subfolder / f"{word_id}.png"

            if img_path.exists():
                records.append((str(img_path), transcription))

    log.info("Parsed %d valid word samples from words.txt", len(records))
    return records


def train_val_split(
    records: List[Tuple[str, str]], val_ratio: float = 0.10, seed: int = 42
) -> Tuple[List, List]:
    """
    Writer-independent split: hold-out all samples from 10 % of forms.
    """
    # Collect unique form IDs (e.g. a01-000u)
    form_ids = sorted({Path(r[0]).parent.name for r in records})
    rng = random.Random(seed)
    rng.shuffle(form_ids)
    n_val = max(1, int(len(form_ids) * val_ratio))
    val_forms = set(form_ids[:n_val])

    train_recs = [r for r in records if Path(r[0]).parent.name not in val_forms]
    val_recs = [r for r in records if Path(r[0]).parent.name in val_forms]
    log.info(
        "Split -> train=%d  val=%d  (val forms=%d)",
        len(train_recs), len(val_recs), len(val_forms),
    )
    return train_recs, val_recs


def preprocess_image(img_path: str, image_size: int = 384) -> Optional[Image.Image]:
    """
    Preprocessing pipeline:
    1. Read as grayscale (handles variability in IAM scan quality)
    2. Adaptive threshold (Otsu) to make ink pop against background
    3. Pad to square, then resize to image_size×image_size
    4. Convert to RGB (TrOCR ViT encoder expects 3 channels)
    """
    img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        return None

    # ── Adaptive thresholding (Otsu) ────────────────────────────────────────
    _, img_thresh = cv2.threshold(
        img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert if background is black (most IAM images are dark-ink on white)
    if img_thresh.mean() < 128:
        img_thresh = cv2.bitwise_not(img_thresh)

    # ── Aspect-ratio preserving pad → square ────────────────────────────────
    h, w = img_thresh.shape
    side = max(h, w)
    canvas = np.ones((side, side), dtype=np.uint8) * 255
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = img_thresh

    # ── Resize ──────────────────────────────────────────────────────────────
    canvas = cv2.resize(canvas, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # ── Grayscale → RGB PIL ─────────────────────────────────────────────────
    pil_img = Image.fromarray(canvas).convert("RGB")
    return pil_img


class IAMDataset(Dataset):
    def __init__(
        self,
        records: List[Tuple[str, str]],
        processor: TrOCRProcessor,
        image_size: int = 384,
        max_label_len: int = 32,
    ):
        self.records = records
        self.processor = processor
        self.image_size = image_size
        self.max_label_len = max_label_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        img_path, label = self.records[idx]

        pil_img = preprocess_image(img_path, self.image_size)
        if pil_img is None:
            # Return a simple white image so DataLoader doesn't crash
            pil_img = Image.new("RGB", (self.image_size, self.image_size), color=255)

        # Processor handles ViT normalization
        pixel_values = self.processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenise label
        labels = self.processor.tokenizer(
            label,
            padding="max_length",
            max_length=self.max_label_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id's with -100 so they are ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# GIT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def git_push_checkpoint(epoch: int, val_cer: float):
    """Commit and push checkpoints + logs to GitHub."""
    if not CONFIG["git_push"]:
        return
    repo = CONFIG["repo_root"]
    try:
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        msg = f"[checkpoint] epoch={epoch}  val_CER={val_cer:.4f}"
        # commit returns exit code 1 if nothing to commit -- that's OK
        commit_result = subprocess.run(
            ["git", "commit", "-m", msg], cwd=repo, capture_output=True
        )
        if commit_result.returncode not in (0, 1):
            raise subprocess.CalledProcessError(commit_result.returncode, "git commit",
                                                stderr=commit_result.stderr)
        subprocess.run(
            ["git", "push", CONFIG["git_remote"], CONFIG["git_branch"]],
            cwd=repo, check=True, capture_output=True,
        )
        log.info("[GIT] Pushed: %s", msg)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        log.warning("Git push failed (non-fatal): %s", stderr[:200])


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, processor, device) -> Tuple[float, float]:
    """Return (avg_loss, CER)."""
    model.eval()
    total_loss = 0.0
    all_preds: List[str] = []
    all_refs: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=_AMP_DEVICE or "cuda", enabled=CONFIG["fp16"]):
                outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()

            # Greedy decode for CER
            generated = model.generate(
                pixel_values,
                max_new_tokens=CONFIG["max_label_len"],
            )
            preds = processor.batch_decode(generated, skip_special_tokens=True)
            refs_ids = labels.clone()
            refs_ids[refs_ids == -100] = processor.tokenizer.pad_token_id
            refs = processor.batch_decode(refs_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    avg_loss = total_loss / max(len(loader), 1)
    val_cer = compute_cer(all_refs, all_preds)
    return avg_loss, val_cer


def train():
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    if device.type == "cuda":
        log.info(
            "GPU: %s  |  VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ── Load processor & model ───────────────────────────────────────────────
    log.info("Loading processor: %s", CONFIG["model_name"])
    processor = TrOCRProcessor.from_pretrained(CONFIG["model_name"])

    log.info("Loading model: %s", CONFIG["model_name"])
    model = VisionEncoderDecoderModel.from_pretrained(CONFIG["model_name"])

    # ── Important decoder config for generation ──────────────────────────────
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = CONFIG["max_label_len"]
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Gradient checkpointing to reduce VRAM
    model.encoder.gradient_checkpointing_enable()

    model.to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    records = parse_words_txt(CONFIG["words_txt"])
    train_recs, val_recs = train_val_split(records, CONFIG["val_split"])

    train_ds = IAMDataset(
        train_recs, processor, CONFIG["image_size"], CONFIG["max_label_len"]
    )
    val_ds = IAMDataset(
        val_recs, processor, CONFIG["image_size"], CONFIG["max_label_len"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    total_steps = (
        math.ceil(len(train_loader) / CONFIG["accum_steps"]) * CONFIG["epochs"]
    )
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if _AMP_DEVICE:
        scaler = GradScaler("cuda", enabled=CONFIG["fp16"])
    else:
        scaler = GradScaler(enabled=CONFIG["fp16"])

    log.info(
        "Training steps: %d  |  Warmup: %d  |  Effective batch: %d",
        total_steps, warmup_steps, CONFIG["batch_size"] * CONFIG["accum_steps"],
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training state ────────────────────────────────────────────────────────
    best_cer = float("inf")
    no_improve = 0
    history: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    # EPOCH LOOP
    # ──────────────────────────────────────────────────────────────────────────
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for step, batch in enumerate(pbar, 1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=_AMP_DEVICE or "cuda", enabled=CONFIG["fp16"]):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / CONFIG["accum_steps"]

            scaler.scale(loss).backward()

            if step % CONFIG["accum_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

            epoch_loss += outputs.loss.item()
            pbar.set_postfix(
                loss=f"{outputs.loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_cer = evaluate(model, val_loader, processor, device)

        log.info(
            "Epoch %d/%d  |  train_loss=%.4f  val_loss=%.4f  val_CER=%.4f",
            epoch, CONFIG["epochs"], avg_train_loss, val_loss, val_cer,
        )

        # Record history
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_cer": round(val_cer, 6),
            }
        )
        with open(CONFIG["log_file"], "w") as f:
            json.dump(history, f, indent=2)

        # ── Save latest checkpoint ────────────────────────────────────────────
        latest_path = ckpt_dir / "latest"
        model.save_pretrained(str(latest_path))
        processor.save_pretrained(str(latest_path))
        log.info("Saved latest checkpoint -> %s", latest_path)

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_cer < best_cer:
            best_cer = val_cer
            no_improve = 0
            best_path = ckpt_dir / "best"
            model.save_pretrained(str(best_path))
            processor.save_pretrained(str(best_path))
            log.info("[BEST] New best CER=%.4f -> saved to %s", best_cer, best_path)
        else:
            no_improve += 1
            log.info(
                "No improvement for %d/%d epochs (best CER=%.4f)",
                no_improve, CONFIG["patience"], best_cer,
            )

        # ── Git push ──────────────────────────────────────────────────────────
        git_push_checkpoint(epoch, val_cer)

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= CONFIG["patience"]:
            log.info(
                "Early stopping triggered after %d epochs without improvement.",
                CONFIG["patience"],
            )
            break

    log.info("Training complete. Best val CER = %.4f", best_cer)
    return best_cer



# ──────────────────────────────────────────────────────────────────────────────
# QUICK INFERENCE (single image)
# ──────────────────────────────────────────────────────────────────────────────
def predict(image_path: str, checkpoint: str = "best") -> str:
    """
    Run inference on a single handwritten word image.

    Parameters
    ----------
    image_path : str  – path to any PNG/JPG word image
    checkpoint : str  – 'best' or 'latest' (or full path to checkpoint dir)
    """
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_path = ckpt_dir / checkpoint if not Path(checkpoint).is_dir() else Path(checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Please run training first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(str(ckpt_path))
    model = VisionEncoderDecoderModel.from_pretrained(str(ckpt_path)).to(device)
    model.eval()

    pil_img = preprocess_image(image_path, CONFIG["image_size"])
    if pil_img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated = model.generate(
            pixel_values,
            max_new_tokens=CONFIG["max_label_len"],
            num_beams=4,
        )

    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrOCR IAM Training / Inference")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="'train' to train the model, 'predict' to run inference on an image",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image for inference (required when --mode predict)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint to use for inference: 'best', 'latest', or full path",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        if args.image is None:
            log.error("--image is required for predict mode")
            sys.exit(1)
        result = predict(args.image, checkpoint=args.checkpoint)
        print(f"\nPredicted text: {result}\n")
