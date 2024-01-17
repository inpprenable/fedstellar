# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 


import logging

import torch

from fedstellar.learning.aggregators.aggregator import Aggregator


class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FedAvg] My config is {}".format(self.config))

    def aggregate(self, models):
        """
        Weighted average of the models.

        Args:
            models: Dictionary with the models (node: model, num_samples).
        """
        if len(models) == 0:
            logging.error("[FedAvg] Trying to aggregate models when there are no models")
            return None

        models = list(models.values())

        # Total Samples
        total_samples = sum(w for _, w in models)

        # Create a Zero Model
        accum = {layer: torch.zeros_like(param) for layer, param in models[-1][0].items()}

        # Add weighted models
        logging.info(f"[FedAvg.aggregate] Aggregating models: num={len(models)}")
        for model, weight in models:
            for layer in accum:
                accum[layer] += model[layer] * weight

        # Normalize Accum
        for layer in accum:
            accum[layer] /= total_samples
            
        # self.print_model_size(accum)

        return accum
