# utils/checkpoint.py
import os
from diffusers import FlaxStableDiffusionPipeline, FlaxPNDMScheduler
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from huggingface_hub import upload_folder
import jax

def save_checkpoint(config, epoch, state, text_encoder_params, vae_params, tokenizer, text_encoder, vae, unet):
    if jax.process_index() != 0:
        return

    os.makedirs(config.output_dir, exist_ok=True)

    safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker", from_pt=True)
    pipeline = FlaxStableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
        safety_checker=safety_checker,
        feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    )

    pipeline.save_pretrained(
        config.output_dir,
        params={
            "text_encoder": jax.device_get(text_encoder_params),
            "vae": jax.device_get(vae_params),
            "unet": jax.device_get(state.params),
            "safety_checker": jax.device_get(safety_checker.params) 
        }
    )

    with open(f"{config.output_dir}/epoch.txt", "w") as f:
        f.write(str(epoch + 1))

    gs_directory = os.getenv('SAVE_DIR')
    with open(f"{config.output_dir}/gcs_save_dir.txt", 'w') as f:
        f.write(f"SAVE TO: {gs_directory}")
    # subprocess.run(f'gsutil -m cp -r ./sd-full-finetuned/* {gs_directory}/full', shell=True)
    # subprocess.run(f'gsutil -m cp *.log {gs_directory}/log', shell=True)

    if config.push_to_hub:
        upload_folder(
            repo_id=config.hub_model_id or config.output_dir,
            folder_path=config.output_dir,
            commit_message=f"Epoch {epoch + 1}",
            token=config.hub_token,
        )
    
