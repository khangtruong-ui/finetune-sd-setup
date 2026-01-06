import jax

if __name__ == "__main__":
    print("===== INITIALIZING =====")
    jax.distributed.initialize()
    print("===== ALL HOSTS CAME IN PLACE =====")

# train.py
from config.training_config import get_config
from training.trainer import Trainer

def main():
    config = get_config()

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
