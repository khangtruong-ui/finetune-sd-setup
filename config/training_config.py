# config/training_config.py
import argparse
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    output_dir: str = "sd-model-finetuned"
    dataset_name: str = None
    train_data_dir: str = None
    resolution: int = 512
    train_batch_size: int = 16
    num_train_epochs: int = 100
    max_train_steps: int = None
    learning_rate: float = 1e-4
    scale_lr: bool = False
    max_grad_norm: float = 1.0
    seed: int = 0
    mixed_precision: str = "no"  # no, fp16, bf16
    push_to_hub: bool = False
    hub_model_id: str = None
    hub_token: str = None
    center_crop: bool = False
    random_flip: bool = False
    max_train_samples: int = None
    from_pt: bool = False
    revision: str = None

def get_config() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Flax Stable Diffusion Fine-tuning")
    # Add all args here (same as before)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--from_pt", action="store_true")
    parser.add_argument("--revision", type=str, default=None)

    args = parser.parse_args()
    return TrainingConfig(**vars(args))
