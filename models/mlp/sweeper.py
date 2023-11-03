import subprocess

from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

if __name__ == "__main__":
    distributions = {
        "learning_rate": loguniform(0.00001, 0.1),
        "hidden_dim": [32, 64, 128, 256],
        "num_hidden_layers": [2, 4, 8, 16],
    }

    sampler = ParameterSampler(distributions, n_iter=10, random_state=42)

    for sample in sampler:
        script = "mlp_training_cli.py"
        args = [
            "--data.batch_size", "500",
            "--model.learning_rate", str(sample['learning_rate']),
            "--model.hidden_dim", str(sample['hidden_dim']),
            "--model.num_hidden_layers", str(sample['num_hidden_layers'])
        ]

        subprocess.run(["python", script, *args])
