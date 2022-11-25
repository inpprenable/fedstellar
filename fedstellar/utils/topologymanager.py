import logging
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class TopologyManager:
    def __init__(
            self, experiment_name, n_nodes, b_symmetric, undirected_neighbor_num=5, topology=None, server=False
    ):
        self.experiment_name = experiment_name
        if topology is None:
            topology = []
        self.n_nodes = n_nodes
        self.b_symmetric = b_symmetric
        self.undirected_neighbor_num = undirected_neighbor_num
        self.server = server
        self.topology = topology
        # Inicialize nodes with array of tuples (0,0,0) with size n_nodes
        self.nodes = np.zeros((n_nodes, 3), dtype=np.int32)

        self.b_fully_connected = False
        if self.undirected_neighbor_num < 2:
            raise ValueError("undirected_neighbor_num must be greater than 2")
        # If the number of neighbors is larger than the number of nodes, then the topology is fully connected
        if self.undirected_neighbor_num >= self.n_nodes - 1 and self.b_symmetric:
            self.b_fully_connected = True

    def draw_graph(self):

        g = nx.from_numpy_array(self.topology)
        # pos = nx.layout.spectral_layout(g)
        # pos = nx.spring_layout(g, pos=pos, iterations=50)
        pos = nx.spring_layout(g, k=0.15, iterations=20)

        fig = plt.figure(dpi=300, figsize=(6, 6), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        # ax.axis('off')
        labels = {}
        color_map = []
        for k in range(self.n_nodes):
            if str(self.nodes[k][2]) == "aggregator":
                color_map.append("orange")
            elif str(self.nodes[k][2]) == "server":
                color_map.append("green")
            else:
                color_map.append("#6182bd")
            if self.nodes[k][3] is not None and self.nodes[k][3] != "127.0.0.1":
                labels[k] = f"N{k}\n" + str(self.nodes[k][3]) + ":" + str(self.nodes[k][1])
            else:
                labels[k] = f"N{k}\n" + str(self.nodes[k][0]) + ":" + str(self.nodes[k][1])
        # nx.draw_networkx_nodes(g, pos_shadow, node_color='k', alpha=0.5)
        nx.draw_networkx_nodes(g, pos, node_color=color_map, linewidths=2)
        nx.draw_networkx_labels(g, pos, labels, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(g, pos, width=2)
        # plt.margins(0.0)
        if self.server:
            plt.scatter([], [], c="green", label='Central Server')
        else:
            plt.scatter([], [], c="orange", label='Aggregator')
        plt.scatter([], [], c="#6182bd", label='Trainer')
        plt.legend()
        # If not exist a folder with the name of the experiment, create it
        import sys
        if not os.path.exists(f"{sys.path[0]}/logs/{self.experiment_name}"):
            os.makedirs(f"{sys.path[0]}/logs/{self.experiment_name}")
        logging.info(f"Saving topology graph to logs/{self.experiment_name}/topology.png")
        plt.savefig(f"{sys.path[0]}/logs/{self.experiment_name}/topology.png", dpi=100, bbox_inches="tight", pad_inches=0)

    def generate_topology(self):
        if self.server:
            self.topology = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
            self.topology[0, :] = 1
            self.topology[:, 0] = 1
            np.fill_diagonal(self.topology, 0)
            return
        if self.b_fully_connected:
            self.__fully_connected()
            return

        if self.b_symmetric:
            self.__randomly_pick_neighbors_symmetric()
        else:
            self.__randomly_pick_neighbors_asymmetric()

    def generate_ring_topology(self, increase_convergence=False):
        self.__ring_topology(increase_convergence=increase_convergence)

    def generate_custom_topology(self, topology):
        self.topology = topology

    def get_topology(self):
        if self.b_symmetric:
            return self.topology
        else:
            return self.topology

    def get_nodes(self):
        return self.nodes

    def add_nodes(self, nodes):
        self.nodes = nodes

    def set_nodes(self, nodes):
        self.nodes = nodes

    def get_node(self, node_idx):
        return self.nodes[node_idx]

    # Get neighbors of a node
    def get_neighbors(self, node_idx):
        neighbors_index = []
        neighbors_data = []
        for i in range(self.n_nodes):
            if self.topology[node_idx][i] == 1:
                neighbors_index.append(i)
                neighbors_data.append(self.nodes[i])

        return neighbors_index, neighbors_data

    def get_neighbors_string(self, node_idx):
        neighbors_index = []
        neighbors_data = []
        for i in range(self.n_nodes):
            if self.topology[node_idx][i] == 1:
                neighbors_index.append(i)
                neighbors_data.append(self.nodes[i])

        neighbors_data_string = ""
        for i in neighbors_data:
            neighbors_data_string += str(i[0]) + ":" + str(i[1]) + " "

        return neighbors_data_string

    def __ring_topology(self, increase_convergence=False):
        topology_ring = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n_nodes, 2, 0)), dtype=np.float32
        )

        if increase_convergence:
            # Create random links between nodes in topology_ring
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if topology_ring[i][j] == 0:
                        if random.random() < 0.1:
                            topology_ring[i][j] = 1
                            topology_ring[j][i] = 1

        np.fill_diagonal(topology_ring, 0)
        self.topology = topology_ring

    def __randomly_pick_neighbors_symmetric(self):
        # First generate a ring topology
        topology_ring = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n_nodes, 2, 0)), dtype=np.float32
        )

        np.fill_diagonal(topology_ring, 0)

        # After, randomly add some links for each node (symmetric)
        # If undericted_neighbor_num is X, then each node has X links to other nodes
        k = int(self.undirected_neighbor_num)
        topology_random_link = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n_nodes, k, 0)), dtype=np.float32
        )

        # generate symmetric topology
        topology_symmetric = topology_ring.copy()
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if topology_symmetric[i][j] == 0 and topology_random_link[i][j] == 1:
                    topology_symmetric[i][j] = topology_random_link[i][j]

        np.fill_diagonal(topology_symmetric, 0)

        self.topology = topology_symmetric

    def __randomly_pick_neighbors_asymmetric(self):
        # randomly add some links for each node (symmetric)
        k = self.undirected_neighbor_num
        topology_random_link = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n_nodes, k, 0)), dtype=np.float32
        )

        np.fill_diagonal(topology_random_link, 0)

        # first generate a ring topology
        topology_ring = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n_nodes, 2, 0)), dtype=np.float32
        )

        np.fill_diagonal(topology_ring, 0)

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if topology_ring[i][j] == 0 and topology_random_link[i][j] == 1:
                    topology_ring[i][j] = topology_random_link[i][j]

        np.fill_diagonal(topology_ring, 0)

        # randomly delete some links
        out_link_set = set()
        for i in range(self.n_nodes):
            len_row_zero = 0
            for j in range(self.n_nodes):
                if topology_ring[i][j] == 0:
                    len_row_zero += 1
            random_selection = np.random.randint(2, size=len_row_zero)
            index_of_zero = 0
            for j in range(self.n_nodes):
                out_link = j * self.n_nodes + i
                if topology_ring[i][j] == 0:
                    if (
                            random_selection[index_of_zero] == 1
                            and out_link not in out_link_set
                    ):
                        topology_ring[i][j] = 1
                        out_link_set.add(i * self.n_nodes + j)
                    index_of_zero += 1

        np.fill_diagonal(topology_ring, 0)

        self.topology = topology_ring

    def __fully_connected(self):
        topology_fully_connected = np.array(
            nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n_nodes, self.n_nodes - 1, 0)),
            dtype=np.float32,
        )

        np.fill_diagonal(topology_fully_connected, 0)

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if topology_fully_connected[i][j] != 1:
                    topology_fully_connected[i][j] = 1

        np.fill_diagonal(topology_fully_connected, 0)

        self.topology = topology_fully_connected
