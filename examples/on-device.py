import os
import time

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


def main():

    config = Config(participant_config_file="/Users/enrique/Documents/PhD/fedstellar/fedstellar/config/participant_config.yaml")

    node = Node(
        MLP(),
        MnistFederatedDM(sub_id=0, number_sub=1, iid=True),
        config=config,
        rol="trainer",
        simulation=True,
    )
    node.start()
    time.sleep(5)
    node.set_start_learning(rounds=10, epochs=5)

    while True:
        time.sleep(1)
        finish = True
        for f in [n.round is None for n in [node]]:
            finish = finish and f

        if finish:
            break

    for n in [node]:
        n.stop()


if __name__ == "__main__":
    main()
