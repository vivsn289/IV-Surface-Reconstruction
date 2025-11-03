import pandas as pd
from pathlib import Path
from typing import Dict, List


class TrainingLogger:
    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{run_name}_log.csv"
        self.history: List[Dict] = []

    def log_epoch(self, epoch_data: Dict):
        self.history.append(epoch_data)

    def save(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_file, index_label="epoch")
