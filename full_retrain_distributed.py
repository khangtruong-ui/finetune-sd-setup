#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from flax import jax_utils
from flax.training import train_state
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed
import subprocess

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
import grain.python as grain

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)

jax.distributed.initialize()
with open('devices.log', 'w') as f:
    mesh = Mesh(np.array(jax.devices()).reshape((-1, 1)), ('data', 'kernel',))
    kernel_sharding = NamedSharding(mesh, P(None, 'kernel'))
    conv_sharding = NamedSharding(mesh, P(None, None, None, 'kernel'))
    sharding = NamedSharding(mesh, P('data'))
    non_sharding = no_sharding = NamedSharding(mesh, P())

    s = f"Mesh: {mesh}\nSharding: {sharding}\nNo sharding: {non_sharding}\nDevices: {jax.local_devices()}"
    f.write(s)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--from_pt",
        action="store_true",
        default=False,
        help="Flag to indicate whether to convert models from PyTorch.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


dataset_name_mapping = {
    "KhangTruong/NWPU_Split": ("image", "raw"),
}


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))

def add_kernel_recursive(unet_weights, prefixes=('', 'attentions', 'transformer_blocks', 'attn', 'to_'), key=jax.random.key(0)):
    unet_weights = jax.tree.map(lambda x: x, unet_weights)
    if len(prefixes) == 0:
        key1, key2 = jax.random.split(key)
        unet_weights['kernel1'] = jax.random.normal(key1, (unet_weights['kernel'].shape[0], 64))
        unet_weights['kernel2'] = jax.random.normal(key2, (64, unet_weights['kernel'].shape[-1]))
        return unet_weights

    # assert any(k.startswith(prefixes[0]) for k in unet_weights.keys()), unet_weights.keys()

    for k in unet_weights.keys():
        key, subkey = jax.random.split(key)
        if k.startswith(prefixes[0]):
            unet_weights[k] = add_kernel_recursive(unet_weights[k], prefixes[1:], subkey)

    return unet_weights

def gradient_mask(grads):

    def mapper(path, x):
        addr = '/'.join(u.key for u in path)
        if 'kernel1' in addr or 'kernel2' in addr:
            return x
        else:
            return jnp.zeros_like(x)

    return jax.tree.map_with_path(mapper, grads)

def merge_kernel(unet_weights, extracted_kernels):
    unet_weights = jax.tree.map(lambda x: x, unet_weights)
    for path in extracted_kernels:
        node = unet_weights

        for k in path.split()[:-1]:
            node = node[k]
        val = extracted_kernels[path]
        assert type(node) is dict
        node[path.split()[-1]] = val
    return unet_weights

def extract_kernel(unet_weights):
    path_val, _ = jax.tree.flatten_with_path(unet_weights)
    path_val = [x for x in path_val if any(('kernel1' in u.key) or ('kernel2' in u.key) for u in x[0])]
    path_val = {' '.join(u.key for u in x[0]): x[1] for x in path_val}
    return path_val

def model_sharding(unet_weights):
    num_devices = jax.device_count()
    
    def mapper(path, x):
        addr = '/'.join(u.key for u in path)
        if ('kernel' in addr or 'kernel' in addr) and x.shape[-1] % num_devices == 0:
            if jnp.ndim(x) == 4:
                return jax.device_put(x, conv_sharding)
            elif jnp.ndim(x) == 2:
                return jax.device_put(x, kernel_sharding)
            else:
                return jax.device_put(x, non_sharding)
        else:
            return jax.device_put(x, non_sharding)

    return jax.tree.map_with_path(mapper, unet_weights)

def random_init(unet_weights):

    class R:
        seed: int = jax.random.key(0)

        def randomize(self, param):
            self.seed, key = jax.random.split(self.seed)
            limit = 1 / jnp.sqrt(640)
            return jax.random.uniform(key, shape=param.shape, minval=-limit, maxval=limit)

    r = R()
    return jax.tree.map(r.randomize, unet_weights)

    
def main():

    ds = load_dataset('KhangTruong/NWPU_Split').with_format('numpy')
    train_ds = ds['train']
    test_ds = ds['test']
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding=True, truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = jnp.array([train_transforms(image) for image in images])
        examples["input_ids"] = tokenize_captions(examples)
        examples['input_ids'] = tokenizer.pad(
            {"input_ids": examples['input_ids']}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="np"
        )

        ret = dict(
            input_ids=examples['input_ids']['input_ids'],
            pixel_values=examples['pixel_values']
        )
        return ret

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)
    sampler = grain.IndexSampler(
        num_records=len(train_dataset),
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(), 
            shard_count=jax.process_count(),
            drop_remainder=True
        ),
        shuffle=True,
        seed=0
    )
    
    # Chain the operations
    loader = grain.DataLoader(
        data_source=train_dataset,
        sampler=sampler,
        operations=[
            grain.Batch(
                batch_size=args.train_batch_size * jax.device_count(),
                drop_remainder=True
            )
        ]
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]

        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        batch = {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
        }
        batch = {k: v.numpy() for k, v in batch.items()}

        return batch

    total_train_batch_size = args.train_batch_size * jax.device_count()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=total_train_batch_size, drop_last=True
    )

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="tokenizer",
    )
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="text_encoder",
        dtype=weight_dtype,
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="vae",
        dtype=weight_dtype,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="unet",
        dtype=weight_dtype,
    )

    # unet_params = add_kernel_recursive(unet_params)
    unet_params = random_init(unet_params)
    unet_params = model_sharding(unet_params)
    
    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Initialize our training
    rng = jax.random.PRNGKey(args.seed)
    train_rngs = rng # jax.random.split(rng, jax.local_device_count())

    @jax.jit
    def train_step(state, text_encoder_params, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(
                batch["input_ids"],
                params=text_encoder_params,
                train=False,
            )[0]

            # Predict the noise residual and compute loss
            model_pred = unet.apply(
                {"params": params}, noisy_latents, timesteps, encoder_hidden_states, train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = (target - model_pred) ** 2
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        print('Tracing train_step done.')
        return new_state, metrics, new_train_rng

    # Create parallel version of the train step
    p_train_step = train_step

    # Replicate the train state on each device
    state = jax.tree.map(lambda x: jax.device_put(x, no_sharding), state)
    text_encoder_params = jax.tree.map(lambda x: jax.device_put(x, no_sharding), text_encoder.params)
    vae_params = jax.tree.map(lambda x: jax.device_put(x, no_sharding), vae_params)

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0

    with open('process.log', 'w') as process_file:
        with open('./sd-full-finetuned/epoch.txt') as f:
            start_epoch = int(f.read())
        epochs = tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch ... ", position=0, file=process_file)
        iter_loader = iter(loader)

        first_time = True
        
        for epoch in epochs:
            # ======================== Training ================================
            train_metrics = []

            steps_per_epoch = len(train_dataset) // total_train_batch_size
            train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False, file=process_file)
            # train
            for _, batch in zip(range(steps_per_epoch), iter_loader):
                unsharded_batch = batch
                batch = jax.tree.map(lambda x: jax.device_put(x, sharding), batch)
                if first_time:
                    first_time = False
                    with open('devices.log', 'a') as f:
                        f.write(f"\n\nArray devices: {jax.tree.map(lambda x: x.device, batch)}\n\nUnsharded array devices: {jax.tree.map(lambda x: x.device, unsharded_batch)}")
                
                state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)
                train_metrics.append(train_metric)

                train_step_progress_bar.update(1)
                train_step_progress_bar.set_postfix({'loss': train_metric["loss"]})

                global_step += 1
                if global_step >= args.max_train_steps:
                    break

            train_step_progress_bar.close()
            epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")
        
            def checkpoint(state):
                # Create the pipeline using using the trained modules and save it.
                if jax.process_index() == 0:
                    scheduler = FlaxPNDMScheduler(
                        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                    )
                    safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
                        "CompVis/stable-diffusion-safety-checker", from_pt=True
                    )
                    pipeline = FlaxStableDiffusionPipeline(
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet,
                        tokenizer=tokenizer,
                        scheduler=scheduler,
                        safety_checker=safety_checker,
                        feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                    )

                    pipeline.save_pretrained(
                        args.output_dir,
                        params={
                            "text_encoder": text_encoder_params,
                            "vae": vae_params,
                            "unet": state.params,
                            "safety_checker": safety_checker.params,
                        },
                    )

                    with open('./sd-full-finetuned/epoch.txt', 'w') as f:
                        f.write(str(epoch + 1))

                    if args.push_to_hub:
                        upload_folder(
                            repo_id=repo_id,
                            folder_path=args.output_dir,
                            commit_message="End of training",
                            ignore_patterns=["step_*", "epoch_*"],

                        )
                    subprocess.run('gsutil -m cp -r ./sd-full-finetuned/* gs://khang-sd-ft/full', shell=True)
                    subprocess.run('gsutil -m cp *.log gs://khang-sd-ft/log', shell=True)
            
            checkpoint(state)


if __name__ == "__main__":
    main()
