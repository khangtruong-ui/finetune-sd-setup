#!/bin/bash
set -e

echo "===== TEST GCS CONNECTION ====="
mkdir -p ./sd-full-finetuned

echo "Test GCS connection" > ./sd-full-finetuned/test.txt

gsutil cp -r ./sd-full-finetuned/test.txt gs://khang-sd-ft/full

gsutil -m cp -r gs://khang-sd-ft/original_weights/* ./sd-full-finetuned

echo "====== PIP INSTALL EVERYTHING ====="

pip install -q jax[tpu] flax optax transformers==4.57.3 datasets diffusers==0.36 torch torchvision Pillow matplotlib grain -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
echo "===== DONE PIP INSTALL EVERYTHING ====="
chmod -R 777 .
# cp attention_flax.py $(pip list -v | grep diffusers | awk '{print $3}')/diffusers/models/attention_flax.py

