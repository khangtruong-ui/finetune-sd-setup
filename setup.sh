#!/bin/bash
set -e


mkdir -p ./sd-full-finetuned
echo "Test GCS connection" > ./sd-full-finetuned/test.txt
gsutil cp -r ./sd-full-finetuned/test.txt gs://khang-sd-ft/full

gsutil -m cp gs://khang-sd-ft/full/* ./sd-full-finetuned

pip install -q jax[tpu] flax optax transformers datasets diffusers==0.36 torch torchvision PIL matplotlib -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

cp attention_flax.py /home/khang/.local/lib/python3.10/site-packages/diffusers/models/attention_flax.py
cp pipeline_flax_stable_diffusion.py ~/.local/lib/python3.11/site-packages/diffusers/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py
