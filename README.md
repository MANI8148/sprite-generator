# Sprite Generator

A from-scratch discrete-latent sprite generator using VQ-VAE + conditional transformer,
compared against SD-LoRA fine-tuning. Trained entirely on free-tier compute.

## Architecture

- **VQ-VAE**: Encodes sprite images into discrete latent codes (256-512 codebook)
- **Transformer Prior**: Small GPT-style model over VQ-VAE token sequences
  conditioned on character class, action, and direction
- **Post-processing**: Native grid-alignment and palette-constrained output

## Stack

- PyTorch for model training
- Kaggle (T4/P100 16GB) for GPU training
- HuggingFace Datasets for data versioning
- HuggingFace Spaces for demo hosting
- GitHub Actions for CI/CD pipeline

## Project Structure

```
sprite-gen/
  data/
    raw/                  # Source sprite packs (gitignored)
    scripts/              # Data pipeline: scrape, clean, caption, push
  models/
    vqvae/                # VQ-VAE encoder/decoder + training
    transformer/          # Conditional transformer prior + training
  kaggle/                 # Kaggle training kernel
  demo/                   # Gradio app for HF Spaces
  eval/                   # Metrics and sample generation
  .github/workflows/      # CI/CD automation
```
