# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 


import logging
import threading


class Aggregator:
    """
    Class to manage the aggregation of models. It is a thread so, aggregation will be done in background if all models were added or timeouts have gone.
    Also, it is an observable so, it will notify the node when the aggregation was done.

    Args:
        node_name: (str): String with the name of the node.
    """

    def __init__(self, node_name="unknown", config=None):
        self.node_name = node_name
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        self.__train_set = []
        self.__waiting_aggregated_model = False
        self.__aggregated_waited_model = False
        self.__models = {}

        # Locks
        self.__agg_lock = threading.Lock()
        self.__finish_aggregation_lock = threading.Lock()

    def aggregate(self, models):
        """
        Aggregate the models.
        """
        print("Not implemented")

    def set_nodes_to_aggregate(self, l):
        """
        List with the name of nodes to aggregate. Be careful, by setting new nodes, the actual aggregation will be lost.

        Args:
            l: List of nodes to aggregate. Empty for no aggregation.

        Raises:
            Exception: If the aggregation is running.
        """
        if not self.__finish_aggregation_lock.locked():
            self.__train_set = l
            self.__models = {}
            self.__finish_aggregation_lock.acquire(timeout=self.config.participant["AGGREGATION_TIMEOUT"])
        else:
            raise Exception(
                "It is not possible to set nodes to aggregate when the aggregation is running."
            )

    def set_waiting_aggregated_model(self, nodes):
        """
        Indicates that the node is waiting for an aggregation. It won't participate in aggregation process.
        The model only will receive a model and then it will be used as an aggregated model.
        """
        self.set_nodes_to_aggregate(nodes)
        self.__waiting_aggregated_model = True

    def clear(self):
        """
        Clear the aggregation (remove train set and release locks).
        """
        self.__agg_lock.acquire()
        self.__train_set = []
        self.__models = {}
        try:
            self.__finish_aggregation_lock.release()
        except:
            pass
        self.__agg_lock.release()

    def get_aggregated_models(self):
        """
        Get the list of aggregated models.

        Returns:
            Name of nodes that collaborated to get the model.
        """
        # Get a list of nodes added
        models_added = [n.split() for n in list(self.__models.keys())]
        # Flatten list
        models_added = [element for sublist in models_added for element in sublist]
        return models_added

    def get_aggregated_models_weights(self):
        # TBD
        # Get a list of weights added
        return self.__models

    def add_model(self, model, contributors, weight):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            contributors: Nodes that collaborated to get the model.
            weight: Number of samples used to get the model.
        """

        nodes = list(contributors)

        # Verify that contributors are not empty
        if contributors == []:
            logging.info(
                f"({self.node_name}) Received a model without a list of contributors."
            )
            self.__agg_lock.release()
            return None

        # Diffusion / Aggregation
        if self.__waiting_aggregated_model and self.__models == {}:
            if set(contributors) == set(self.__train_set):
                logging.info(f"({self.node_name}) Received an aggregated model because all contributors are in the train set (me too). Overwriting my model with the aggregated model.")
                self.__models = {}
                self.__models = {" ".join(nodes): (model, 1)}
                self.__waiting_aggregated_model = False
                self.__finish_aggregation_lock.release()
                return contributors

        else:
            self.__agg_lock.acquire()

            # Check if aggregation is needed
            if len(self.__train_set) > len(self.get_aggregated_models()):
                # Check if all nodes are in the train_set
                if all([n in self.__train_set for n in nodes]):
                    logging.info(f'({self.node_name}) All contributors are in the train set. Adding model.')
                    # Check if the model is a full/partial aggregation
                    if len(nodes) == len(self.__train_set):
                        logging.info(f'({self.node_name}) The number of contributors is equal to the number of nodes in the train set. --> Full aggregation.')
                        self.__models = {}
                        self.__models[" ".join(nodes)] = (model, weight)
                        logging.info(
                            f"({self.node_name}) Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                        )
                        # Finish agg
                        self.__finish_aggregation_lock.release()
                        # Unlock and Return
                        self.__agg_lock.release()
                        return self.get_aggregated_models()

                    elif all([n not in self.get_aggregated_models() for n in nodes]):
                        logging.info(f'({self.node_name}) All contributors are not in the aggregated models. --> Partial aggregation.')
                        # Aggregate model
                        self.__models[" ".join(nodes)] = (model, weight)
                        logging.info(
                            f"({self.node_name}) Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                        )

                        # Check if all models were added
                        if len(self.get_aggregated_models()) >= len(self.__train_set):
                            logging.info(
                                f"({self.node_name}) All models were added. Finishing aggregation."
                            )
                            self.__finish_aggregation_lock.release()

                        # Unlock and Return
                        self.__agg_lock.release()
                        return self.get_aggregated_models()

                    else:
                        logging.info(
                            f"({self.node_name}) Can't add a model that has already been added {nodes}"
                        )
                else:
                    logging.info(
                        f"({self.node_name}) Can't add a model from a node ({nodes}) that is not in the training test."
                    )
            else:
                logging.info(
                    f"({self.node_name}) Received a model when is not needed."
                )
            self.__agg_lock.release()
            return None

    def wait_and_get_aggregation(self):
        """
        Wait for aggregation to finish.

        Returns:
            Aggregated model.

        Raises:
            Exception: If waiting for an aggregated model and several models were received.
        """
        timeout = self.config.participant["AGGREGATION_TIMEOUT"]
        # Wait for aggregation to finish (then release the lock again)
        self.__finish_aggregation_lock.acquire(timeout=timeout)
        try:
            self.__finish_aggregation_lock.release()
        except:
            pass

        # If awaiting an aggregated model, return it
        if self.__waiting_aggregated_model:
            if len(self.__models) == 1:
                return list(self.__models.values())[0][0]
            elif len(self.__models) == 0:
                logging.info(
                    f"({self.node_name}) Timeout reached by waiting for an aggregated model. Continuing with the local model."
                )
            raise Exception(
                f"Waiting for an an aggregated but several models were received: {self.__models.keys()}"
            )
        # Start aggregation
        logging.info(f'({self.node_name}) Starting aggregation.')
        n_model_aggregated = sum(
            [len(nodes.split()) for nodes in list(self.__models.keys())]
        )

        # Timeout / All models
        if n_model_aggregated != len(self.__train_set):
            logging.info(
                f"({self.node_name}) Aggregating models, timeout reached. Missing models: {set(self.__train_set) - set(self.__models.keys())}"
            )
        else:
            logging.info(f"({self.node_name}) Aggregating models.")

        # Notify node
        return self.aggregate(self.__models)

    def get_partial_aggregation(self, except_nodes):
        """
        Obtain a partial aggregation.

        Args:
            except_nodes (list): List of nodes to exclude from the aggregation.

        Returns:
            Aggregated model, nodes aggregated and aggregation weight.
        """
        dict_aux = {}
        nodes_aggregated = []
        aggregation_weight = 0
        models = self.__models.copy()
        for n, (m, s) in list(models.items()):
            spplited_nodes = n.split()
            if all([n not in except_nodes for n in spplited_nodes]):
                dict_aux[n] = (m, s)
                nodes_aggregated += spplited_nodes
                aggregation_weight += s

        # If there are no models to aggregate
        if len(dict_aux) == 0:
            return None, None, None

        logging.info(
            f"({self.node_name}) Aggregating models: dict_aux={dict_aux.keys()}"
        )

        return (self.aggregate(dict_aux), nodes_aggregated, aggregation_weight)
