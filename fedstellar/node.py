# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


import logging
import math
import random
import threading
import time

from pytorch_lightning.loggers import WandbLogger

from fedstellar.base_node import BaseNode
from fedstellar.communication_protocol import CommunicationProtocol
from fedstellar.config.config import Config
from fedstellar.learning.aggregators.fedavg import FedAvg
from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.pytorch.lightninglearner import LightningLearner
from fedstellar.role import Role
from fedstellar.utils.observer import Events, Observer


class Node(BaseNode):
    """
    Class based on a base node that allows **p2p Federated Learning**.

    Metrics will be saved under a folder with the name of the node.

    Args:
        model: Model to be learned. Careful, model should be compatible with data and the learner.
        data: Dataset to be used in the learning process. Careful, model should be compatible with data and the learner.
        host (str): Host where the node will be listening.
        port (int): Port where the node will be listening.
        learner (NodeLearner): Learner to be used in the learning process. Default: LightningLearner.
        simulation (bool): If False, node will share metrics and communication will be encrypted. Default: True.

    Attributes:
        round (int): Round of the learning process.
        totalrounds (int): Total number of rounds of the learning process.
        learner (NodeLearner): Learner to be used in the learning process.
        aggregator (Aggregator): Aggregator to be used in the learning process.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(
            self,
            idx,
            experiment_name,
            model,
            data,
            host="127.0.0.1",
            port=None,
            config=Config,
            learner=LightningLearner,
            role=None,
            simulation=True,
    ):
        # Super init
        BaseNode.__init__(self, experiment_name, host, port, simulation, config)
        Observer.__init__(self)

        self.idx = idx
        logging.debug("[NODE] My idx is {}".format(self.idx))

        # Import configuration file
        self.config = config

        # Learning
        self.round = None
        self.totalrounds = None
        self.__model_initialized = False
        self.__initial_neighbors = []
        self.__start_thread_lock = threading.Lock()

        # Learner and learner logger
        # log_model="all" to log model
        # mode="disabled" to disable wandb
        logging.getLogger("wandb").setLevel(logging.ERROR)
        wandblogger = WandbLogger(project="framework-enrique", group=self.experiment_name, name=self.get_name(), mode="disabled")
        wandblogger.watch(model, log="all")
        import wandb
        img_topology = wandb.Image(f"logs/{self.experiment_name}/topology.png", caption="Topology")
        if self.idx == 0:
            wandblogger.log_image(key="topology", images=[img_topology])
        self.learner = learner(model, data, logger=wandblogger, log_name=self.get_name())

        self.role = role
        logging.info("[NODE] Role: " + str(self.role))

        # Aggregator
        self.aggregator = FedAvg(node_name=self.get_name(), config=self.config, role=self.role)
        self.aggregator.add_observer(self)

        # Train Set Votes
        self.__train_set = []
        self.__train_set_votes = {}
        self.__train_set_votes_lock = threading.Lock()

        # Locks
        self.__wait_votes_ready_lock = threading.Lock()
        self.__finish_aggregation_lock = threading.Lock()
        self.__finish_aggregation_lock.acquire()
        self.__wait_init_model_lock = threading.Lock()
        self.__wait_init_model_lock.acquire()

    #########################
    #    Node Management    #
    #########################

    def connect_to(self, h, p, full=False, force=False):
        """
        Connects a node to another. If learning is running, connections are not allowed (it should be forced).
        Careful, if connection is forced with a new node, it will produce timeouts in the network.

        Args:
            h (str): The host of the node.
            p (int): The port of the node.
            full (bool): If True, the node will be connected to the entire network.
            force (bool): If True, the node will be connected even though it should not be.

        Returns:
            node: The node that has been connected to.
        """
        # Check if learning is running
        if self.round is not None and not force:
            logging.info(
                "[NODE] Cant connect to other nodes when learning is running."
            )
            return None

        # Connect
        return super().connect_to(h, p, full, force)

    def stop(self):
        """
        Stop the node and the learning if it is running.
        """
        if self.round is not None:
            self.__stop_learning()
        self.learner.close()
        super().stop()

    ##########################
    #    Learning Setters    #
    ##########################

    def set_data(self, data):
        """
        Set the data to be used in the learning process (learner).

        Args:
            data: Dataset to be used in the learning process.
        """
        self.learner.set_data(data)

    def set_model(self, model):
        """
        Set the model to use.
        Carefully, model, not only weights.

        Args:
            model: Model to be learned.
        """
        self.learner.set_model(model)

    ###############################################
    #         Network Learning Management         #
    ###############################################

    def set_start_learning(self, rounds=1, epochs=1):
        """
        Start the learning process in the entire network.
        If the node is not the "starting" node, it will wait for the starting node to start the learning process.

        NOTE:
        In first instance, this functionality only is used in one node of the network (defined by the controller).
        After that, the other participants can start this functionality when they receive Events.START_LEARNING_EVENT

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        if self._terminate_flag.is_set():
            logging.info(
                "[NODE] Node must be running to start learning"
            )
            return
        if self.round is None:
            # Start Learning
            logging.info("[NODE] I am the initializer node... | Broadcasting START_LEARNING | Rounds: {} | Epochs: {}".format(rounds, epochs))
            self.broadcast(
                CommunicationProtocol.build_start_learning_msg(rounds, epochs)
            )
            # Initialize model
            self.broadcast(CommunicationProtocol.build_model_initialized_msg())
            self.__wait_init_model_lock.release()
            self.__model_initialized = (
                True  # esto seguramente sobre, con locks es suficiente
            )
            # Learning Thread
            self.__start_learning_thread(rounds, epochs)
        else:
            logging.info("[NODE] Learning already started")

    def set_stop_learning(self):
        """
        Stop the learning process in the entire network.
        """
        if self.round is not None:
            self.broadcast(CommunicationProtocol.build_stop_learning_msg())
            self.__stop_learning()
        else:
            logging.info("[NODE] Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning_thread(self, rounds, epochs):
        learning_thread = threading.Thread(
            target=self.__start_learning, args=(rounds, epochs)
        )
        learning_thread.name = "learning_thread-" + self.get_name()
        learning_thread.daemon = True
        learning_thread.start()

    def __start_learning(self, rounds, epochs):
        """
        Start the learning process in the local node.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        self.__start_thread_lock.acquire()  # Used to avoid create duplicated training threads
        if self.round is None:
            self.round = 0
            self.totalrounds = rounds
            self.learner.init()
            self.__start_thread_lock.release()

            begin = time.time()

            # Send the model parameters (initial model) to neighbors
            self.__gossip_model_difusion(initialization=True)

            # Wait to guarantee new connection heartbeats convergence and fix neighbors
            wait_time = self.config.participant_config["WAIT_HEARTBEATS_CONVERGENCE"] - (time.time() - begin)
            if wait_time > 0:
                time.sleep(wait_time)
            # TODO: Check this parameter
            self.__initial_neighbors = (
                self.get_neighbors()
            )  # used to restore the original list of neighbors after the learning round

            logging.info("[NODE.__start_learning] Learning started in node {} -> Round: {} | Epochs: {}".format(self.get_name(), self.round, epochs))
            self.learner.set_epochs(epochs)
            self.learner.create_trainer()
            self.__train_step()

    def __stop_learning(self):
        """
        Stop the learning process in the local node. Interrupts learning process if it's running.
        """
        logging.info("[NODE] Stopping learning")
        # Rounds
        self.round = None
        self.totalrounds = None
        # Leraner
        self.learner.interrupt_fit()
        # Aggregator
        self.aggregator.check_and_run_aggregation(force=True)
        self.aggregator.set_nodes_to_aggregate([])
        self.aggregator.clear()
        # Try to free wait locks
        try:
            self.__wait_votes_ready_lock.release()
        except threading.ThreadError:
            pass

    ####################################
    #         Model Aggregation         #
    ####################################

    def add_model(self, m):
        """
        Add a model. If the model isn't inicializated, the recieved model is used for it. Otherwise, the model is aggregated using the **aggregator**.

        Args:
            m: Encoded model. Contains model and their contributors
        """
        # Check if Learning is running
        if self.round is not None:
            try:
                if self.__model_initialized:
                    # Add model to aggregator
                    (
                        decoded_model,
                        contributors,
                        weight,
                    ) = self.learner.decode_parameters(m)
                    logging.info("[NODE.add_model] Model received from {}, now I add the model using self.aggregator.add_model()".format(contributors))
                    if self.learner.check_parameters(decoded_model):
                        models_added = self.aggregator.add_model(
                            decoded_model, contributors, weight
                        )
                        if models_added is not None:
                            # TODO: Fix bug at MacBook. When CPU is high, only new nodes will be sent.
                            self.broadcast(
                                CommunicationProtocol.build_models_aggregated_msg(
                                    models_added
                                )
                            )
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    # Initialize model
                    model, _, _ = self.learner.decode_parameters(m)
                    self.learner.set_parameters(model)
                    self.__model_initialized = True
                    logging.info("[NODE] Initialization Model Weights")
                    self.__wait_init_model_lock.release()
                    self.broadcast(CommunicationProtocol.build_model_initialized_msg())

            except DecodingParamsError as e:
                logging.error("[NODE] Error decoding parameters: " + str(e))
                self.stop()

            except ModelNotMatchingError as e:
                logging.error("[NODE] Models not matching: " + str(e))
                self.stop()

            except Exception as e:
                logging.error("[NODE] Error adding model: " + str(e))
                self.stop()
                raise e
        else:
            logging.error(
                "[NODE] Tried to add a model while learning is not running"
            )

    #######################
    #    Training Steps    #
    #######################

    def __train_step(self):
        """
        Train the model in the local node.
        If the node is in the __train_set list, the training is performed. Otherwise, the node waits for the training to be performed by another node.
        Returns:

        """

        # Set train set
        if self.round is not None:
            # self.__train_set = self.__vote_train_set()
            for n in self.get_neighbors():
                if n.get_name() not in self.__train_set:
                    self.__train_set.append(n.get_name())
            self.__train_set.append(self.get_name()) if self.get_name() not in self.__train_set else None
            # if self.role != Role.TRAINER:
            #     self.__train_set.append(self.get_name()) if self.get_name() not in self.__train_set else None
            # else:
            #     self.__train_set.remove(self.get_name()) if self.get_name() in self.__train_set else None
            logging.info("[NODE.__train_step] __train_set = {}".format(self.__train_set))
            self.__validate_train_set()

        # Si estoy en el train set:
        #   1. Me conecto a los vecinos
        #   2. Evalúo el modelo (la primera vez es con el modelo recién inicializado)
        #   3. Entreno el modelo
        #   4. Agrego el modelo
        #       4.1 aggregator.add_model(
        #           modelo to add
        #           nodos que colaboran para obtener el modelo = número de modelos necesarios para agregar
        #           número de muestras para obtener el modelo
        #           ) --> es una funcion padre que llama, en este caso, a fedavg
        #       4.2 self.broadcast(
        #             CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
        #           ) --> notifico MODELS_AGGREGATED (con mi nombre como parámetro) a los vecinos que tengo un modelo agregado y posteriormente lo envío
        #           --> los que reciben el mensaje únicamente añaden a __models_aggregated los nodos agregados
        #       4.3 __gossip_model_aggregation()
        #       4.4 __gossip_model_difusion()
        #       4.5 __on_round_finished()

        # Si es servidor o agregador, siempre va a entrar aquí
        # TODO: Improve in the future
        # is_train_set = self.get_name() in self.__train_set
        is_train_set = True
        if is_train_set and self.role != Role.TRAINER:

            # Full connect train set
            if self.round is not None:
                self.__connect_and_set_aggregator()

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None and self.role != Role.SERVER:  # El participante servidor no entrena (en CFL)
                self.__train()

            # Aggregate Model
            if self.round is not None:
                # Add my model to aggregator. This will trigger the aggregation
                # Proceso: recogida de modelos de los vecinos
                # Objetivo: obtener un modelo agregado
                # Parámetros de la función:
                #   - Parámetros del modelo
                #   - Nodos que colaboran para obtener el modelo (yo)
                #   - Número de muestras (len(train_dataset)) para obtener el modelo

                # En concreto, añado mi modelo a la lista de modelos agregados
                # Esto es necesario para poder enviar el modelo local a los vecinos
                self.aggregator.add_model(
                    self.learner.get_parameters(),
                    [self.get_name()],
                    self.learner.get_num_samples()[0],
                )
                # Notifico a los vecinos que tengo un modelo agregado y posteriormente lo envío
                # Los que reciben el mensaje únicamente añaden a __models_aggregated los nodos agregados
                #   self.__models_aggregated = list(set(models + self.__models_aggregated))
                self.broadcast(
                    CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
                )

                self.__gossip_model_aggregation()
        elif self.role == Role.TRAINER:
            logging.info("[NODE.__train_step] Role.TRAINER process...")
            if self.round is not None:
                self.__connect_and_set_aggregator()

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()

            # Aggregate Model
            if self.round is not None:
                self.aggregator.add_model(
                    self.learner.get_parameters(),
                    [self.get_name()],
                    self.learner.get_num_samples()[0],
                )

                self.broadcast(
                    CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
                )

                self.__gossip_model_aggregation()

                self.aggregator.set_waiting_aggregated_model()

        else:
            # Role.IDLE functionality

            # Set Models To Aggregate
            # Node won't participate in aggregation process.
            # __waiting_aggregated_model = True
            # Then, when the node receives a PARAMS_RECEIVED_EVENT, it will run add_model, and it set parameters to the model
            self.aggregator.set_waiting_aggregated_model()

        # Gossip aggregated model
        if self.round is not None:
            self.__gossip_model_difusion()

        # Finish round
        if self.round is not None:
            self.__on_round_finished()

    ################
    #    Voting    #
    ################

    def __vote_train_set(self):

        # Vote
        candidates = self.get_network_nodes()  # al least himself
        logging.debug(
            "[NODE] Candidates to train set: {}".format(candidates)
        )
        if self.get_name() not in candidates:
            candidates.append(self.get_name())

        # Send vote
        samples = min(self.config.participant_config["TRAIN_SET_SIZE"], len(candidates))
        nodes_voted = random.sample(candidates, samples)
        weights = [
            math.floor(random.randint(0, 1000) / (i + 1)) for i in range(samples)
        ]
        votes = list(zip(nodes_voted, weights))
        logging.info("[NODE.__vote_train_set] Voting for train set: {}".format(votes))

        # Adding votes
        self.__train_set_votes_lock.acquire()
        self.__train_set_votes[self.get_name()] = dict(votes)
        self.__train_set_votes_lock.release()

        # Send and wait for votes
        logging.info("[NODE] Sending train set vote.")
        logging.debug("[NODE] Self Vote: {}".format(votes))
        self.broadcast(
            CommunicationProtocol.build_vote_train_set_msg(self.get_name(), votes)
        )
        logging.debug("[NODE] Waiting other node votes.")

        # Get time
        count = 0
        begin = time.time()

        while True:
            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info(
                    "[NODE] Stopping on_round_finished process."
                )
                return []

            # Update time counters (timeout)
            count = count + (time.time() - begin)
            begin = time.time()
            timeout = count > self.config.participant_config["VOTE_TIMEOUT"]

            # Clear non candidate votes
            self.__train_set_votes_lock.acquire()
            nc_votes = {
                k: v for k, v in self.__train_set_votes.items() if k in candidates
            }
            self.__train_set_votes_lock.release()

            # Determine if all votes are received
            votes_ready = set(candidates) == set(nc_votes.keys())
            if votes_ready or timeout:

                if timeout and not votes_ready:
                    logging.info(
                        "[NODE] Timeout for vote aggregation. Missing votes from {}".format(
                            set(candidates) - set(nc_votes.keys())
                        )
                    )

                results = {}
                for node_vote in list(nc_votes.values()):
                    for i in range(len(node_vote)):
                        k = list(node_vote.keys())[i]
                        v = list(node_vote.values())[i]
                        if k in results:
                            results[k] += v
                        else:
                            results[k] = v

                # Order by votes and get TOP X
                results = sorted(
                    results.items(), key=lambda x: x[0], reverse=True
                )  # to equal solve of draw
                results = sorted(results, key=lambda x: x[1], reverse=True)
                top = min(len(results), self.config.participant_config["TRAIN_SET_SIZE"])
                results = results[0:top]
                results = {k: v for k, v in results}
                votes = list(results.keys())
                logging.info("[NODE.__vote_train_set] Final results (train set): {}".format(votes))

                # Clear votes
                self.__train_set_votes = {}
                logging.info(
                    "[NODE] Computed {} votes.".format(len(nc_votes))
                )
                return votes

            # Wait for votes or refresh every 2 seconds
            self.__wait_votes_ready_lock.acquire(timeout=2)

    def __validate_train_set(self):
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        for tsn in self.__train_set:
            if tsn not in self.get_network_nodes():
                if tsn != self.get_name():
                    self.__train_set.remove(tsn)

        logging.info(
            "[NODE.__validate_train_set] Train set of {} nodes: {}".format(
                len(self.__train_set), self.__train_set
            )
        )

    ##########################
    #    Connect Trainset    #
    ##########################

    def __connect_and_set_aggregator(self):
        # Set Models To Aggregate
        self.aggregator.set_nodes_to_aggregate(self.__train_set)
        logging.info("[NODE.__connect_and_set_aggregator] Aggregator set to: {}".format(self.__train_set))
        for node in self.__train_set:
            if node != self.get_name():
                h, p = node.split(":")
                if p.isdigit():
                    nc = self.get_neighbor(h, int(p))
                    # If the node is not connected, connect it (to avoid duplicated connections only a node connects to the other)
                    if nc is None and self.get_name() > node:
                        self.connect_to(h, int(p), force=True)
                else:
                    logging.info(
                        "[NODE] Node {} has an invalid port".format(
                            node.split(":")
                        )
                    )

        # Wait connections
        count = 0
        begin = time.time()
        while True:
            count = count + (time.time() - begin)
            if count > self.config.participant_config["TRAIN_SET_CONNECT_TIMEOUT"]:
                logging.info("[NODE] Timeout for train set connections.")
                break
            if (len(self.__train_set) == len(
                    [
                        nc
                        for nc in self.get_neighbors()
                        if nc.get_name() in self.__train_set
                    ]
            ) + 1
            ):
                break
            time.sleep(0.1)

    ############################
    #    Train and Evaluate    #
    ############################

    def __train(self):
        logging.info("[NODE] Training...")
        self.learner.fit()

    def __evaluate(self):
        logging.info("[NODE] Evaluating...")
        results = self.learner.evaluate()
        if results is not None:
            logging.info(
                "[NODE] Evaluated. Losss: {}, Metric: {}".format(
                    results[0], results[1]
                )
            )
            # Send metrics
            if not self.simulation:
                logging.info(
                    "[NODE] Broadcasting metrics.".format(
                        len(self.get_neighbors())
                    )
                )
                encoded_msgs = CommunicationProtocol.build_metrics_msg(
                    self.get_name(), self.round, results[0], results[1]
                )
                self.broadcast(encoded_msgs)

    ######################
    #    Round finish    #
    ######################

    def __on_round_finished(self):
        # Remove trainset connections
        for nc in self.get_neighbors():
            if nc not in self.__initial_neighbors:
                self.rm_neighbor(nc)
        # Set Next Round
        self.aggregator.clear()
        self.learner.finalize_round()  # TODO: Fix to improve functionality
        self.round = self.round + 1
        # Clear node aggregation
        for nc in self.get_neighbors():
            nc.clear_models_aggregated()

        # Next Step or Finish
        logging.info(
            "[NODE] Round {} of {} finished.".format(
                self.round, self.totalrounds
            )
        )
        if self.round < self.totalrounds:
            self.__train_step()
        else:
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            self.round = None
            self.totalrounds = None
            self.__model_initialized = False
            logging.info(
                "[NODE] FL experiment finished".format(
                    self.round, self.totalrounds
                )
            )

    #########################
    #    Model Gossiping    #
    #########################

    def __gossip_model_aggregation(self):
        # Anonymous functions
        candidate_condition = lambda nc: nc.get_name() in self.__train_set and len(nc.get_models_aggregated()) < len(self.__train_set)
        status_function = lambda nc: (nc.get_name(), len(nc.get_models_aggregated()))
        model_function = lambda nc: self.aggregator.get_partial_aggregation(nc.get_models_aggregated())

        # Gossip
        logging.info("[NODE.__gossip_model_aggregation] Gossiping model aggregation...")
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model_difusion(self, initialization=False):
        # Send model parameters using gossiping
        # Wait a model (init or aggregated)
        if initialization:
            logging.info("[NODE.__gossip_model_difusion] Initialization=True")
            self.__wait_init_model_lock.acquire()
            candidate_condition = lambda nc: not nc.get_model_initialized()
        else:
            logging.info("[NODE.__gossip_model_difusion] Initialization=False")
            self.__finish_aggregation_lock.acquire()
            candidate_condition = lambda nc: nc.get_model_ready_status() < self.round

        # Anonymous functions
        status_function = lambda nc: nc.get_name()
        model_function = lambda _: (
            self.learner.get_parameters(),
            None,
            None,
        )  # At diffusion, contributors are not relevant

        # Gossip
        logging.info("[NODE.__gossip_model_difusion] Gossiping model parameters...")
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model(self, candidate_condition, status_function, model_function):
        logging.info("[NODE.__gossip_model] Traceback", stack_info=True)
        # Initialize list with status of nodes in the last X iterations
        last_x_status = []
        j = 0

        while True:
            # Get time to calculate frequency
            begin = time.time()

            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info(
                    "[NODE] Stopping model gossip process.")
                return

            # Get nodes which need models
            logging.info("[NODE.__gossip_model] Neighbors: {}".format(self.get_neighbors()))
            for nc in self.get_neighbors():
                logging.info("[NODE.__gossip_model] Neighbor: {} | My __train_set: {} | Nc.modelsaggregated: {}".format(nc, self.__train_set, nc.get_models_aggregated()))
                logging.info("[NODE.__gossip_model] Neighbor: {} | Candidate_condition return: {}".format(nc, candidate_condition(nc)))
                logging.info("[NODE.__gossip_model] Neighbor: {} | Status_function return: {}".format(nc, status_function(nc)))

            nei = [nc for nc in self.get_neighbors() if candidate_condition(nc)]
            logging.info("[NODE.__gossip_model] Selected based on condition: {}".format(self.get_neighbors(), nei))

            # Determine end of gossip
            if not nei:
                logging.info("[NODE] Gossip finished.")
                return

            # Save state of neighbors. If nodes are not responding gossip will stop
            if len(last_x_status) != self.config.participant_config["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]:
                last_x_status.append([status_function(nc) for nc in nei])
            else:
                last_x_status[j] = str([status_function(nc) for nc in nei])
                j = (j + 1) % self.config.participant_config["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]

                # Check if las messages are the same
                for i in range(len(last_x_status) - 1):
                    if last_x_status[i] != last_x_status[i + 1]:
                        break
                    logging.info(
                        "[NODE] Gossiping exited for {} equal rounds.".format(
                            self.config.participant_config["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]
                        )
                    )
                    return

            # Select a random subset of neighbors
            samples = min(self.config.participant_config["GOSSIP_MODELS_PER_ROUND"], len(nei))
            nei = random.sample(nei, samples)

            # Generate and Send Model Partial Aggregations (model, node_contributors)
            for nc in nei:
                model, contributors, weights = model_function(nc)
                # Send Partial Aggregation
                if model is not None:
                    logging.info(
                        "[NODE] Gossiping model to {}.".format(
                            nc.get_name()
                        )
                    )
                    encoded_model = self.learner.encode_parameters(
                        params=model, contributors=contributors, weight=weights
                    )
                    logging.info("[NODE.__gossip_model] Building params message | Contributors: {}".format(contributors))
                    encoded_msgs = CommunicationProtocol.build_params_msg(encoded_model, self.config.participant_config["BLOCK_SIZE"])
                    # Send Fragments
                    for msg in encoded_msgs:
                        nc.send(msg)
                else:
                    logging.info("[NODE.__gossip_model] Model returned by model_function is None")
            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1 / self.config.participant_config["GOSSIP_MODELS_FREC"] - time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)

    ###########################
    #     Observer Events     #
    ###########################

    def update(self, event, obj):
        """
        Observer update method. Used to handle events that can occur in the different components and connections of the node.

        Args:
            event (Events): Event that has occurred.
            obj: Object that has been updated.
        """
        if len(str(obj)) > 300:
            logging.info("[NODE.update (observer)] Event that has occurred: {} | Obj information: Too long [...]".format(event))
        else:
            logging.info("[NODE.update (observer)] Event that has occurred: {} | Obj information: {}".format(event, obj))

        if event == Events.NODE_CONNECTED_EVENT:
            n, force = obj
            if self.round is not None and not force:
                logging.info(
                    "[NODE] Cant connect to other nodes when learning is running. (however, other nodes can be connected to the node.)"
                )
                n.stop()
                return

        elif event == Events.SEND_ROLE_EVENT:
            self.broadcast(CommunicationProtocol.build_role_msg(self.get_name(), self.role))

        elif event == Events.ROLE_RECEIVED_EVENT:
            # Update the heartbeater with the role node
            # obj = (node_name, role)
            self.heartbeater.add_node_role(obj[0], obj[1])

        elif event == Events.AGGREGATION_FINISHED_EVENT:
            # Set parameters and communate it to the training process
            if obj is not None:
                logging.info("[NODE.update] Override the local model with obj received")
                self.learner.set_parameters(obj)
                # Share that aggregation is done
                self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))
            else:
                logging.error(
                    "[NODE] Aggregation finished with no parameters"
                )
                self.stop()
            try:
                self.__finish_aggregation_lock.release()
            except threading.ThreadError:
                pass

        elif event == Events.START_LEARNING_EVENT:
            self.__start_learning_thread(obj[0], obj[1])

        elif event == Events.STOP_LEARNING_EVENT:
            self.__stop_learning()

        elif event == Events.PARAMS_RECEIVED_EVENT:
            self.add_model(obj)

        elif event == Events.METRICS_RECEIVED_EVENT:
            name, round, loss, metric = obj
            self.learner.log_validation_metrics(loss, metric, round=round, name=name)

        elif event == Events.TRAIN_SET_VOTE_RECEIVED_EVENT:
            node, votes = obj
            self.__train_set_votes_lock.acquire()
            self.__train_set_votes[node] = votes
            self.__train_set_votes_lock.release()
            # Communicate to the training process that a vote has been received
            try:
                self.__wait_votes_ready_lock.release()
            except threading.ThreadError:
                pass

        # Execute BaseNode update
        super().update(event, obj)
