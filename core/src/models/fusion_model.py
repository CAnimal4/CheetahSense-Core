import numpy as np


class RuleBasedFusionModel:
    """
    Minimal fusion placeholder that averages probability vectors.
    """

    def __init__(self, num_models: int = 1):
        self.num_models = num_models

    def fuse(self, prob_vectors: list[np.ndarray]) -> np.ndarray:
        if not prob_vectors:
            raise ValueError("No probability vectors provided.")
        stacked = np.stack(prob_vectors, axis=0)
        return np.mean(stacked, axis=0)
