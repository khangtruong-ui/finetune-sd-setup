# train.py
from config.training_config import get_config
from training.trainer import Trainer
from utils.logging import setup_logging

def main():
    config = get_config()
    setup_logging()

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    import jax
    jax.distributed.initialize()
    main()
