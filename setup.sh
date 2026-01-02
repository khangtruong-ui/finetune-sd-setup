#!/bin/bash
set -e

echo "===== TEST GCS CONNECTION ====="
mkdir -p ./sd-full-finetuned
echo "Test GCS connection" > ./sd-full-finetuned/test.txt
gsutil cp -r ./sd-full-finetuned/test.txt gs://khang-sd-ft/full

gsutil -m cp -r gs://khang-sd-ft/full/* ./sd-full-finetuned

echo "====== PIP INSTALL EVERYTHING ====="

pip install -q jax[tpu] flax optax transformers datasets diffusers==0.36 torch torchvision Pillow matplotlib -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

cp attention_flax.py /home/$USER/.local/lib/python3.10/site-packages/diffusers/models/attention_flax.py
