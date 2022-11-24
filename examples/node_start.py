import logging
import os
import sys
import time

# Add the path to the fedstellar folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fedstellar.config.config import Config
from fedstellar.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from fedstellar.learning.pytorch.mnist_examples.models.mlp import MLP
from fedstellar.node import Node



os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# This script can be called with a different arguments
# 1. Host IP
# 2. Host Port
# 3. Neighbors IP
# parser = argparse.ArgumentParser(description='Distributed node')
# parser.add_argument("ip", help="Host IP")
# parser.add_argument("port", help="Host Port")
# parser.add_argument("neighbors", help="Neighbors")
# args = parser.parse_args()

def main_process(ip, port, neighbors):
    # IP
    print(ip)
    # Port
    print(port)
    # Neighbors
    print(neighbors)


def main():
    idx = int(sys.argv[1])
    experiment_name = sys.argv[2]
    ip = sys.argv[3]
    port = int(sys.argv[4])
    ipdemo = sys.argv[5]
    n_nodes = int(sys.argv[6])
    start_node = sys.argv[7] == "True"
    role = sys.argv[8]
    neighbors = sys.argv[9:]

    config = Config(participant_config_file="/Users/enrique/Documents/PhD/fedstellar/fedstellar/config/participant_config.yaml")

    node = Node(
        idx,
        experiment_name,
        MLP(),
        MnistFederatedDM(sub_id=idx, number_sub=n_nodes, iid=True),
        hostdemo=ipdemo,
        host=ip,
        port=port,
        config=config,
        role=role,
        simulation=True,
    )

    node.start()
    time.sleep(1)

    # Node Connection
    for i in neighbors:
        node.connect_to(i.split(':')[0], int(i.split(':')[1]), full=False)
        time.sleep(1)

    logging.info(f"Neighbors: {node.get_neighbors()}")
    logging.info(f"Network nodes: {node.get_network_nodes()}")

    time.sleep(1)

    if start_node:
        node.set_start_learning(rounds=10, epochs=5)


if __name__ == "__main__":
    # print(args.ip)
    # print(args.port)
    # print(args.neighbors)
    main()
