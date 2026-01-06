# data/dataset.py
import random
from torchvision import transforms
import grain.python as grain
from datasets import load_dataset
import jax.numpy as jnp
import jax

def get_transforms(config):
    return transforms.Compose([
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
        transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def get_dataloader(config, tokenizer):
    if config.dataset_name:
        dataset = load_dataset(config.dataset_name)
    else:
        dataset = load_dataset("imagefolder", data_files={"train": f"{config.train_data_dir}/**"})

    if config.max_train_samples:
        dataset["train"] = dataset["train"].shuffle(seed=config.seed).select(range(config.max_train_samples))

    transform = get_transforms(config)

    def preprocess(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        examples["pixel_values"] = jnp.array([transform(img) for img in images])

        captions = examples['raw']
        tokens = tokenizer(captions, padding="max_length", truncation=True, max_length=tokenizer.model_max_length)
        padded_tokens = tokenizer.pad(
            {"input_ids": tokens.input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="np"
        )
        examples["input_ids"] = padded_tokens
        ret = dict(
            input_ids=examples['input_ids'],
            pixel_values=examples['pixel_values']
        )
        return ret

    dataset = dataset["train"].with_transform(preprocess)

    sampler = grain.IndexSampler(
        num_records=len(dataset),
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=True,
        ),
        shuffle=True,
        seed=config.seed,
    )

    loader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[grain.Batch(batch_size=config.train_batch_size * jax.local_device_count(), drop_remainder=True)],
    )

    loader_length = len(dataset) // (config.train_batch_size * jax.device_count())
    return {'loader': loader, 'length': loader_length}
