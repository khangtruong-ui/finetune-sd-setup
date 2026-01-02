#!/bin/bash
set -e

echo "===== TEST GCS CONNECTION ====="
mkdir -p ./sd-full-finetuned
chmod -R 777 .
echo "Test GCS connection" > ./sd-full-finetuned/test.txt
gsutil cp -r ./sd-full-finetuned/test.txt gs://khang-sd-ft/full

gsutil -m cp -r gs://khang-sd-ft/full/* ./sd-full-finetuned

echo "====== PIP INSTALL EVERYTHING ====="

pip install -q jax[tpu] flax optax transformers datasets diffusers==0.36 torch torchvision Pillow matplotlib -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c "import jax; print(jax.device_count())"
python -c "import torch, diffusers;"
cp attention_flax.py $(pip list -v | grep diffusers | awk '{print $3}')/diffusers/models/attention_flax.py

