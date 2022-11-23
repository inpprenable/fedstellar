import sys

from fedstellar.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from fedstellar.learning.pytorch.mnist_examples.models.mlp import MLP
from fedstellar.node import Node

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 nodo2.py <self_host> <self_port>")
        sys.exit(1)

    node = Node(
        MLP(),
        MnistFederatedDM(sub_id=0, number_sub=2),
        host=sys.argv[1],
        port=int(sys.argv[2]),
    )
    node.start()
