# training/train_step.py
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from diffusers import FlaxDDPMScheduler


def create_train_step(unet, vae, noise_scheduler):
    """
    Creates a compiled train_step function for Stable Diffusion fine-tuning.
    """
    noise_scheduler_state = noise_scheduler.create_state()

    def train_step(state: train_state.TrainState,
                   vae_params,
                   text_encoder_params,
                   batch,
                   rng):
        """
        Single training step: computes loss and applies gradients.
        """
        dropout_rng, noise_rng = jax.random.split(rng, 2)

        def compute_loss(params):
            # Encode images to latents
            latents_dist = vae.apply(
                {"params": vae_params},
                batch["pixel_values"],
                deterministic=True,
                method=vae.encode,
            ).latent_dist
            latents = latents_dist.sample(noise_rng)
            latents = jnp.transpose(latents, (0, 3, 1, 2)) * vae.config.scaling_factor

            # Sample noise and timesteps
            noise = jax.random.normal(noise_rng, latents.shape)
            timesteps = jax.random.randint(
                noise_rng,
                (latents.shape[0],),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(
                noise_scheduler_state, latents, noise, timesteps
            )

            # Get text embeddings
            encoder_hidden_states = text_encoder_params
            # Actually we pass the output directly in trainer, so adjust accordingly

            # In practice, we pre-compute text embeddings outside if frozen
            # But here we assume text_encoder_params is the full frozen param tree

            # Predict noise residual
            model_pred = unet.apply(
                {"params": params},
                noisy_latents,
                timesteps,
                batch["encoder_hidden_states"],  # precomputed in trainer
                train=True,
            ).sample

            # MSE loss (epsilon prediction)
            target = noise
            loss = ((model_pred - target) ** 2).mean()
            return loss

        loss, grads = jax.value_and_grad(compute_loss)(state.params)
        new_state = state.apply_gradients(grads=grads)

        return new_state, loss, dropout_rng

    return jax.jit(train_step)
