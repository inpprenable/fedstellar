import logging
import os
import sys
import time

from fedstellar.learning.pytorch.femnist.femnist import FEMNISTDataModule

# Add the path to the fedstellar folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fedstellar.config.config import Config
from fedstellar.learning.pytorch.mnist.mnist import MNISTDataModule
from fedstellar.learning.pytorch.mnist.models.mlp import MLP
from fedstellar.learning.pytorch.femnist.models.cnn import CNN as CNN_femnist
from fedstellar.node import Node

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def main():
    idx = int(sys.argv[1])

    config = Config(entity="participant", participant_config_file=f"config/participant_{idx}.json")
    n_nodes = len(config.participant["network_args"]["neighbors"].split()) + 1
    experiment_name = config.participant["scenario_args"]["name"]
    model = config.participant["model_args"]["model"]
    if model == "MLP":
        model = MLP()
    elif model == "CNN":
        model = CNN_femnist()
    else:
        raise ValueError(f"Model {model} not supported")

    dataset = config.participant["data_args"]["dataset"]
    if dataset == "MNIST":
        dataset = MNISTDataModule(sub_id=idx, number_sub=n_nodes, iid=True)
    elif dataset == "FEMNIST":
        dataset = FEMNISTDataModule(sub_id=idx, number_sub=n_nodes, root_dir="data")
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    hostdemo = config.participant["network_args"]["ipdemo"]
    host = config.participant["network_args"]["ip"]
    port = config.participant["network_args"]["port"]

    neighbors = config.participant["network_args"]["neighbors"].split()

    node = Node(
        idx=idx,
        experiment_name=experiment_name,
        model=model,
        data=dataset,
        hostdemo=hostdemo,
        host=host,
        port=port,
        config=config,
        encrypt=False
    )

    node.start()
    time.sleep(1)

    # Node Connection
    for i in neighbors:
        print(f"Connecting to {i}")
        node.connect_to(i.split(':')[0], int(i.split(':')[1]), full=False)
        time.sleep(1)

    logging.info(f"Neighbors: {node.get_neighbors()}")
    logging.info(f"Network nodes: {node.get_network_nodes()}")

    time.sleep(1)

    start_node = config.participant["device_args"]["start"]

    if start_node:
        node.set_start_learning(rounds=10, epochs=5)


if __name__ == "__main__":
    os.system("clear")
    main()
