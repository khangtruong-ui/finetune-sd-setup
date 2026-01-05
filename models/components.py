# models/components.py
import jax.numpy as jnp
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel

def load_models(config):
    weight_dtype = {"no": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}[config.mixed_precision]

    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer", revision=config.revision, from_pt=config.from_pt
    )

    text_encoder = FlaxCLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", dtype=weight_dtype,
        revision=config.revision, from_pt=config.from_pt
    )

    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="vae", dtype=weight_dtype,
        revision=config.revision, from_pt=config.from_pt
    )

    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", dtype=weight_dtype,
        revision=config.revision, from_pt=config.from_pt
    )

    return tokenizer, text_encoder, vae, vae_params, unet, unet_params
