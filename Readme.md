# Hand-Written-Text-reg

Handwritten Text Recognition using **TrOCR** (Transformer-based OCR) trained on the **IAM Handwriting Database**.

---

## Architecture

| Component | Detail |
|---|---|
| Model | `microsoft/trocr-small-handwritten` |
| Encoder | Vision Transformer (ViT-Small) |
| Decoder | GPT-2 (causal language model) |
| Loss | Cross-Entropy (via `VisionEncoderDecoderModel`) |
| Metric | Character Error Rate (CER) |

## Hardware Target

| Spec | Value |
|---|---|
| GPU | NVIDIA RTX 3050 |
| VRAM | 4 GB |
| Precision | fp16 (mixed precision) |
| Effective batch | 16 (batch=2 × accum=8) |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train_trocr.py --mode train
```

- Checkpoints saved to `checkpoints/best/` (lowest val CER) and `checkpoints/latest/`
- Training log (loss & CER per epoch) saved to `training_log.json`
- Every checkpoint is automatically committed and pushed to GitHub

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 20 (early stopping patience=5) |
| Learning rate | 5e-5 |
| Scheduler | Cosine annealing with 5% warmup |
| Optimizer | AdamW (weight decay=0.01) |
| Image size | 384×384 |
| Max label len | 32 tokens |

---

## Inference

```bash
# Single word image
python inference.py --image path/to/word.png

# Use latest checkpoint instead of best
python inference.py --image path/to/word.png --checkpoint latest
```

Or via the training script directly:

```bash
python train_trocr.py --mode predict --image path/to/word.png
```

---

## Dataset

- **IAM Handwriting Database** — word-level segmented images
- Labels: `words.txt` (word ID → transcription mapping)
- Images: `words/{folder}/{subfolder}/{word_id}.png`
- Split: 90% train / 10% val (writer-independent by form ID)

---

## Preprocessing Pipeline

1. Grayscale conversion
2. Otsu adaptive thresholding (ink-background separation)
3. Aspect-ratio preserving pad to square
4. Resize to 384×384
5. RGB conversion + ViT normalization (via `TrOCRProcessor`)

---

## Project Structure

```
Hand-Written-Text-reg/
├── train_trocr.py       ← Main training + inference script
├── inference.py         ← Standalone inference CLI
├── requirements.txt     ← Python dependencies
├── training_log.json    ← Per-epoch loss & CER (auto-generated)
├── train.log            ← Full training log (auto-generated)
├── checkpoints/
│   ├── best/            ← Best model (lowest val CER)
│   └── latest/          ← Most recent epoch
└── SimpleHTR/           ← Legacy HTR implementation
```
