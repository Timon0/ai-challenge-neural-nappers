import subprocess

from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

if __name__ == "__main__":
    distributions = {
        "learning_rate": loguniform(1e-7, 1e-5),
        "encoder_layer_nhead": [1, 2],
        "num_layers": [2, 4],
    }

    sampler = ParameterSampler(distributions, n_iter=10, random_state=42)

    for sample in sampler:
        script = "main.py"
        args = [
            "--model.learning_rate", str(sample['learning_rate']),
            "--model.encoder_layer_nhead", str(sample['encoder_layer_nhead']),
            "--model.num_layers", str(sample['num_layers'])
        ]

        subprocess.run(["python", script, *args])
