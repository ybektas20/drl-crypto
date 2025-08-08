# DRL-Crypto

Deep reinforcement learning for cryptocurrency portfolio management using EIIE (Ensemble of Identical Independent Evaluators) algorithm.

## Quick Start with Docker

1. **Build the container:**
```bash
docker build -t drl-crypto .
```

2. **Run the complete pipeline:**
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results drl-crypto python main.py
```

This will:
- Download crypto data from Binance
- Process and generate features  
- Train the portfolio optimization model
- Save results to `./results/`

## Local Setup

If you prefer to run without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

## Configuration

Edit config files in `configs/`:
- `data.yaml` - symbols, date ranges, frequencies
- `train.yaml` - model hyperparameters

## What it does

- Downloads aggregate trade data for 7 cryptocurrencies
- Trains a CNN to predict optimal portfolio weights
- Supports long-only and long-short strategies
- Includes transaction costs and risk management
