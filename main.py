import os
import torch
from src.training.mappo_trainer import MAPPOTrainer
from src.utils.config import Config
from src.utils.logger import AGLLogger

def main():
    # 1. Load configuration
    config = Config()
    config.print_config()

    # 2. Initialize Logger
    logger = AGLLogger(log_dir=config.LOG_DIR)
    logger.log_config(config)

    # 3. Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 4. Initialize and train the MAPPO trainer
    # The trainer now handles its own visualization and internal logging
    trainer = MAPPOTrainer(config)
    
    print("\n--- Starting Elite Training Pipeline ---")
    try:
        trainer.train(num_episodes=config.NUM_EPISODES)
        
        # 5. Save the trained models
        trainer.save_models(config.MODEL_SAVE_PATH)
        print(f"\nTraining complete. Models saved to {config.MODEL_SAVE_PATH}")
        print(f"Visualization results are in {config.RESULTS_DIR}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
        trainer.save_models(config.MODEL_SAVE_PATH.replace(".pth", "_interrupted.pth"))

if __name__ == "__main__":
    main()
