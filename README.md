
# to build the image:
docker build -t drl-crypto:1.0 .

# to run the image:
docker run --rm -it \
  drl-crypto:1.0 \
  bash

# sample data loading command:
python scripts/download_binance.py \
    --symbols BTCUSDT ETHUSDT ADAUSDT BNBUSDT DOGEUSDT SOLUSDT XRPUSDT \
    --start 2025-05-01 \
    --end   2025-05-31 \
    --dest  data/raw/aggTrades