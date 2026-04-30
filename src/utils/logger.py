import json
import os
import time
from datetime import datetime

class AGLLogger:
    """
    Professional logger for the Autonomous Green Lung (AGL) project.
    Handles experiment tracking, metric logging, and model metadata.
    """
    def __init__(self, log_dir="logs/"):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(self.log_dir, f"exp_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.jsonl")
        self.config_file = os.path.join(self.experiment_dir, "config.json")
        
        print(f"AGL Logger initialized. Experiment logs at {self.experiment_dir}")

    def log_config(self, config_obj):
        """Logs the configuration used for the experiment."""
        config_dict = {attr: getattr(config_obj, attr) for attr in dir(config_obj) 
                       if not attr.startswith("__") and not callable(getattr(config_obj, attr))}
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def log_metrics(self, episode, metrics_dict):
        """Logs a dictionary of metrics for a given episode."""
        log_entry = {
            "timestamp": time.time(),
            "episode": episode,
            **metrics_dict
        }
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

    def save_summary(self, summary_dict):
        """Saves a final summary of the experiment."""
        summary_file = os.path.join(self.experiment_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=4)
