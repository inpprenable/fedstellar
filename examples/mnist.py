import time

from fedstellar.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from fedstellar.learning.pytorch.mnist_examples.models.mlp import MLP
from fedstellar.node import Node


def mnist_execution(n, start, simulation, conntect_to=None, iid=True):
    # This function is used to execute the MNIST example with n nodes (simulation)
    # In a real scenario, the nodes are executed in different machines
    # - Controller creates the topology, send via Mender the code to the nodes
    # - Each node executed the code and connects to the other nodes (neighbors in the topology, at first)

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(
            MLP(),
            MnistFederatedDM(sub_id=i, number_sub=n, iid=iid),
            rol="trainer",
            simulation=simulation,
        )
        node.start()
        nodes.append(node)

    # Connect other network
    if conntect_to is not None:
        nodes[0].connect_to(conntect_to[0], conntect_to[1])

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect_to(nodes[i].host, nodes[i].port, full=True)
        time.sleep(1)

    time.sleep(5)
    print("Starting...")

    for n in nodes:
        print(len(n.get_neighbors()))
        print(len(n.get_network_nodes()))

    # Start Learning
    if start:
        nodes[0].set_start_learning(rounds=5, epochs=2)
    else:
        time.sleep(20)

    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

    for node in nodes:
        node.stop()


if __name__ == "__main__":
    mnist_execution(3, True, True)
    # for _ in range(50):
    #    mnist_execution(3, True, True)
    #    break
