#!/bin/bash
set -e

echo "===== TEST GCS CONNECTION ====="
mkdir -p ./sd-full-finetuned
chmod -R 777 .
echo "Test GCS connection" > ./sd-full-finetuned/test.txt

arg=$(echo "$1" | xargs)

echo "Reset argument: '$arg'"

if [ "$arg" != "reset" ]; then 
    gsutil cp -r ./sd-full-finetuned/test.txt gs://khang-sd-ft/full
else
    gsutil cp -r ./sd-full-finetuned/test.txt gs://khang-sd-ft/original_weights
fi
gsutil -m cp -r gs://khang-sd-ft/full/* ./sd-full-finetuned

echo "====== PIP INSTALL EVERYTHING ====="

pip install -q jax[tpu] flax optax transformers==4.57.3 datasets diffusers==0.36 torch torchvision Pillow matplotlib grain -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
echo "===== DONE PIP INSTALL EVERYTHING ====="
# cp attention_flax.py $(pip list -v | grep diffusers | awk '{print $3}')/diffusers/models/attention_flax.py

