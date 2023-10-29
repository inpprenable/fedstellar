# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
import json
import logging
import math
import os
from datetime import datetime

from fedstellar.attacks.aggregation import create_attack
from fedstellar.learning.aggregators.aggregator import create_malicious_aggregator
from fedstellar.learning.aggregators.helper import cosine_similarity
from fedstellar.learning.pytorch.remotelogger import FedstellarWBLogger
from fedstellar.learning.pytorch.statisticsloggerv2 import FedstellarLogger
from fedstellar.messages import LearningNodeMessages
from fedstellar.proto import node_pb2
from fedstellar.role import Role

os.environ['WANDB_SILENT'] = 'true'

# Import the requests module
import requests

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import random
import threading
import time

from lightning.pytorch.loggers import CSVLogger

from fedstellar.base_node import BaseNode
from fedstellar.config.config import Config
from fedstellar.learning.aggregators.fedavg import FedAvg
from fedstellar.learning.aggregators.krum import Krum
from fedstellar.learning.aggregators.median import Median
from fedstellar.learning.aggregators.trimmedmean import TrimmedMean
from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.pytorch.lightninglearner import LightningLearner


class Node(BaseNode):
    """
    Class based on a base node that allows **FEDERATED LEARNING**.

    Metrics will be saved under a folder with the name of the node.

    Args:
        model: Model to be learned. Careful, model should be compatible with data and the learner.
        data: Dataset to be used in the learning process. Careful, model should be compatible with data and the learner.
        host (str): Host where the node will be listening.
        port (int): Port where the node will be listening.
        learner (NodeLearner): Learner to be used in the learning process. Default: LightningLearner.
        simulation (bool): If True, the node will be simulated. Default: True.
        encrypt (bool): If True, node will encrypt the communications. Default: False.

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
            hostdemo=None,
            host="127.0.0.1",
            port=None,
            config=Config,
            learner=LightningLearner,
            encrypt=False,
            model_poisoning=False,
            poisoned_ratio=0,
            noise_type='gaussian',
    ):
        # Super init
        BaseNode.__init__(self, experiment_name, hostdemo, host, port, encrypt, config)

        self.idx = idx
        logging.info("[NODE] My idx is {}".format(self.idx))

        # Import configuration file
        self.config = config

        # Report the configuration to the controller (first instance)
        self.__report_status_to_controller()
        self.__start_reporter_thread()

        # Add message handlers
        self.add_message_handler(
            LearningNodeMessages.START_LEARNING, self.__start_learning_callback
        )
        self.add_message_handler(
            LearningNodeMessages.REPUTATION, self.__reputation_callback
        )
        self.add_message_handler(
            LearningNodeMessages.STOP_LEARNING, self.__stop_learning_callback
        )
        self.add_message_handler(
            LearningNodeMessages.MODEL_INITIALIZED, self.__model_initialized_callback
        )
        self.add_message_handler(
            LearningNodeMessages.VOTE_TRAIN_SET, self.__vote_train_set_callback
        )
        self.add_message_handler(
            LearningNodeMessages.MODELS_AGGREGATED, self.__models_agregated_callback
        )
        self.add_message_handler(
            LearningNodeMessages.MODELS_READY, self.__models_ready_callback
        )
        self.add_message_handler(LearningNodeMessages.METRICS, self.__metrics_callback)

        # Learning
        self.round = None
        self.totalrounds = None
        self.__train_set = []
        self.__models_aggregated = {}
        self.__nei_status = {}

        # Attack environment
        self.model_dir = self.config.participant['tracking_args']["model_dir"]
        self.model_name = f"{self.model_dir}/participant_{self.idx}_model.pk"
        self.model_poisoning = model_poisoning
        self.poisoned_ratio = poisoned_ratio
        self.noise_type = noise_type

        self.__trusted_nei = []
        self.__is_malicious = False

        # with reputation mechansim
        # Todo: change to initial function
        self.with_reputation = self.config.participant['defense_args']["with_reputation"]

        self.is_dynamic_topology = self.config.participant['defense_args']["is_dynamic_topology"]
        self.is_dynamic_aggregation = self.config.participant['defense_args']["is_dynamic_aggregation"]

        logging.info(f"[NODE] with_reputation {self.with_reputation}")
        logging.info(f"[NODE] is_dynamic_topology {self.is_dynamic_topology}")
        logging.info(f"[NODE] is_dynamic_aggregation {self.is_dynamic_aggregation}")

        # Learner and learner logger
        # log_model="all" to log model
        # mode="disabled" to disable wandb
        if self.config.participant['tracking_args']['enable_remote_tracking']:
            logging.info("[NODE] Tracking W&B enabled")
            logging.getLogger("wandb").setLevel(logging.ERROR)
            if self.hostdemo:
                wandblogger = FedstellarWBLogger(project="platform-enrique", group=self.experiment_name,
                                                 name=f"participant_{self.idx}")
            else:
                wandblogger = FedstellarWBLogger(project="platform-enrique", group=self.experiment_name,
                                                 name=f"participant_{self.idx}")
            wandblogger.watch(model, log="all")
            self.learner = learner(model, data, config=self.config, logger=wandblogger)
        else:
            if self.config.participant['tracking_args']['local_tracking'] == 'csv':
                logging.info("[NODE] Tracking CSV enabled")
                csvlogger = CSVLogger(f"{self.log_dir}", name="metrics", version=f"participant_{self.idx}")
                self.learner = learner(model, data, config=self.config, logger=csvlogger)
            elif self.config.participant['tracking_args']['local_tracking'] == 'web':
                logging.info("[NODE] Tracking Web enabled")
                tensorboardlogger = FedstellarLogger(f"{self.log_dir}", name="metrics",
                                                     version=f"participant_{self.idx}", log_graph=True)
                self.learner = learner(model, data, config=self.config, logger=tensorboardlogger)

        logging.info("[NODE] Role: " + str(self.config.participant["device_args"]["role"]))
        if self.config.participant["adversarial_args"]["attacks"] != "No Attack":
            self.__is_malicious = True
        logging.info("[NODE] Benign node" if not self.__is_malicious else "[NODE] Malicious node")

        # Aggregators
        if self.config.participant["aggregator_args"]["algorithm"] == "FedAvg":
            self.aggregator = FedAvg(node_name=self.get_name(), config=self.config)
        elif self.config.participant["aggregator_args"]["algorithm"] == "Krum":
            self.aggregator = Krum(node_name=self.get_name(), config=self.config)
        elif self.config.participant["aggregator_args"]["algorithm"] == "Median":
            self.aggregator = Median(node_name=self.get_name(), config=self.config)
        elif self.config.participant["aggregator_args"]["algorithm"] == "TrimmedMean":
            self.aggregator = TrimmedMean(node_name=self.get_name(), config=self.config)

        # whether to use the dynamic aggregator
        if self.is_dynamic_aggregation:
            # Target Aggregators
            if self.config.participant["defense_args"]["target_aggregation"] == "FedAvg":
                self.target_aggregation = FedAvg(node_name=self.get_name(), config=self.config)
            elif self.config.participant["defense_args"]["target_aggregation"] == "Krum":
                self.target_aggregation = Krum(node_name=self.get_name(), config=self.config)
            elif self.config.participant["defense_args"]["target_aggregation"] == "Median":
                self.target_aggregation = Median(node_name=self.get_name(), config=self.config)
            elif self.config.participant["defense_args"]["target_aggregation"] == "TrimmedMean":
                self.target_aggregation = TrimmedMean(node_name=self.get_name(), config=self.config)

        # Train Set Votes
        self.__train_set_votes = {}
        self.__train_set_votes_lock = threading.Lock()

        # Locks
        self.__start_thread_lock = threading.Lock()
        self.__wait_votes_ready_lock = threading.Lock()
        self.__model_initialized_lock = threading.Lock()
        self.__model_initialized_lock.acquire()
        self.finish_round_lock = threading.Lock()

    ######################
    #    Msg Handlers    #
    ######################

    def __start_learning_callback(self, msg):
        self.__start_learning_thread(int(msg.args[0]), int(msg.args[1]))

    def __disrupt_connection_using_reputation(self, malicious_nodes):
        # Disrupt the connection with the malicious nodes
        malicious_nodes = list(set(malicious_nodes) & set(self.get_neighbors()))
        # logging.info(f"({self.addr}) Received reputation from {msg.source} with malicious nodes {malicious_nodes}")
        logging.info(f"Disrupting connection with malicious nodes at round {self.learner.get_round()}")
        logging.info(f"({self.addr}) Removing {malicious_nodes} from {self.get_neighbors()}")
        logging.info(f"get neighbors before aggregation at round {self.learner.get_round()}: {self.get_neighbors()}")
        for malicious_node in malicious_nodes:
            if (self.get_name() != malicious_node) and (malicious_node not in self.__trusted_nei):
                self._neighbors.remove(malicious_node)
        logging.info(f"get neighbors after aggregation at round {self.learner.get_round()}: {self.get_neighbors()}")

        self.__connect_with_benign(malicious_nodes)

    def __connect_with_benign(self, malicious_nodes):
        # Define the thresholds for estabilish new connections, if len(neighbors) <= lower_threshold, trigger to build new connections
        # until len(neighbors) reached the higher_threshold or connected with all benign nodes
        lower_threshold = 1
        higher_threshold = len(self.__train_set) - 1

        # make sute higher_threshold is not lower than lower_threshold
        if higher_threshold < lower_threshold:
            higher_threshold = lower_threshold

        benign_nodes = [i for i in self.__train_set if i not in malicious_nodes]
        logging.info(
            f"({self.addr})__reputation_callback benign_nodes at round {self.learner.get_round()}: {benign_nodes}")
        if len(self.get_neighbors()) <= lower_threshold:
            for node in benign_nodes:
                if len(self.get_neighbors()) <= higher_threshold and self.get_name() != node:
                    connected = self.connect(node)
                    if connected:
                        logging.info(
                            f"({self.addr}) Connect new connection with at round {self.learner.get_round()}: {connected}")

    def __reputation_callback(self, msg):
        # receive malicious nodes information from neighbors
        malicious_nodes = msg.args  # List of malicious nodes
        if self.with_reputation:
            if len(malicious_nodes) > 0 and not self.__is_malicious:
                if self.is_dynamic_topology:
                    self.__disrupt_connection_using_reputation(malicious_nodes)

                if self.is_dynamic_aggregation and self.aggregator != self.target_aggregation:
                    self.__dynamic_aggergator(self.aggregator.get_aggregated_models_weights(), malicious_nodes)

    def __dynamic_aggergator(self, aggregated_models_weights, malicious_nodes):
        logging.info(f"malicious detected at round {self.learner.get_round()}, change aggergation protocol!")
        if self.aggregator != self.target_aggregation:
            logging.info(f"get_aggregated_models current aggregator is: {self.aggregator}")
            # do some threading
            self.aggregator = self.target_aggregation
            self.aggregator.set_nodes_to_aggregate(self.__train_set)

            # current_models = {}
            for subnodes in aggregated_models_weights.keys():
                sublist = subnodes.split()
                (submodel, weights) = aggregated_models_weights[subnodes]
                for node in sublist:
                    if node not in malicious_nodes:
                        self.aggregator.add_model(
                            submodel, [node], weights, source=self.get_name(), round=self.round
                        )
            logging.info(f"get_aggregated_models current aggregator is: {self.aggregator}")

    def __stop_learning_callback(self, _):
        self.__stop_learning()

    def __model_initialized_callback(self, msg):
        self.__nei_status[msg.source] = -1

    def __vote_train_set_callback(self, msg):
        # check moment: round or round + 1 because of node async
        ########################################################
        ### try to improve clarity in message moment check
        ########################################################
        if msg.round in [self.round, self.round + 1]:
            # build vote dict
            votes = msg.args
            tmp_votes = {}
            for i in range(0, len(votes), 2):
                tmp_votes[votes[i]] = int(votes[i + 1])
            # set votes
            self.__train_set_votes_lock.acquire()
            self.__train_set_votes[msg.source] = tmp_votes
            self.__train_set_votes_lock.release()
            # Communicate to the training process that a vote has been received
            try:
                self.__wait_votes_ready_lock.release()
            except:
                pass
        else:
            logging.error(
                f"({self.addr}) Vote received in a late round. Ignored. {msg.round} != {self.round} / {self.round + 1}"
            )

    def __models_agregated_callback(self, msg):
        if msg.round == self.round:
            self.__models_aggregated[msg.source] = msg.args

    def __models_ready_callback(self, msg):
        ########################################################
        # try to improve clarity in message moment check
        ########################################################
        if msg.round in [self.round - 1, self.round]:
            self.__nei_status[msg.source] = int(msg.args[0])
        else:
            logging.error(
                f"({self.addr}) Models ready in a late round. Ignored. {msg.round} != {self.round} / {self.round - 1}"
            )

    def __metrics_callback(self, msg):
        name = msg.source
        round = msg.round
        loss, metric = msg.args[0:2]
        loss = float(loss)
        self.learner.log_validation_metrics(loss, metric, round=round, name=name)

    ############################
    #  GRPC - Remote Services  #
    ############################

    def add_model(self, request, _):
        """
        GRPC service. It is called when a node wants to add a model to the network.
        """
        # Check if Learning is running
        if self.round is not None:
            # Check source
            # self.round is modified at the end of the round, so we need to check if the message is from the previous round
            # implement a lock to avoid concurrent access to round
            self.finish_round_lock.acquire()
            current_round = self.round
            self.finish_round_lock.release()
            if request.round != current_round:
                logging.info(
                    f"({self.addr}) add_model (gRPC) | Model Reception in a late round ({request.round} != {self.round})."
                )
                return node_pb2.ResponseMessage()

            # Check moment (not init and invalid round)
            if (
                    not self.__model_initialized_lock.locked()
                    and len(self.__train_set) == 0
            ):
                logging.info(
                    f"({self.addr}) add_model (gRPC) | Model Reception when there is no trainset"
                )
                return node_pb2.ResponseMessage()

            try:
                if not self.__model_initialized_lock.locked():
                    # Add model to aggregator
                    logging.info(f"({self.addr}) add_model (gRPC) | Remote Service using gRPC (executed by {request.source})")
                    decoded_model = self.learner.decode_parameters(request.weights)
                    if self.learner.check_parameters(decoded_model):
                        models_added = self.aggregator.add_model(
                            decoded_model, request.contributors, request.weight, source=request.source, round=request.round
                        )
                        if models_added is not None:
                            logging.info(
                                f'({self.addr}) add_model (gRPC) | Models added using local aggregator, now sending models_added using MODELS_AGGREGATED): {models_added}'
                            )
                            # Communicate Aggregation
                            self._neighbors.broadcast_msg(
                                self._neighbors.build_msg(
                                    LearningNodeMessages.MODELS_AGGREGATED, models_added
                                )
                            )
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    # Initialize model (try to handle concurrent initializations)
                    logging.info(f"({self.addr}) add_model (gRPC) | Initializing model (executed by {request.source})")
                    try:
                        self.__model_initialized_lock.release()
                        model = self.learner.decode_parameters(request.weights)
                        self.learner.set_parameters(model)
                        logging.info(f"({self.addr}) add_model (gRPC) | Model Weights Initialized")
                        # Communicate Initialization
                        self._neighbors.broadcast_msg(
                            self._neighbors.build_msg(
                                LearningNodeMessages.MODEL_INITIALIZED
                            )
                        )
                    except RuntimeError:
                        # unlock unlocked lock
                        pass

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError as e:
                logging.error(f"({self.addr}) add_model (gRPC) | Error decoding parameters.")
                self.stop()

            except ModelNotMatchingError as e:
                logging.error(f"({self.addr}) add_model (gRPC) | Models not matching.")
                self.stop()

            except Exception as e:
                logging.error(f"({self.addr}) add_model (gRPC) | Unknown error adding model: {e}")
                self.stop()

        else:
            logging.debug(
                f"({self.addr}) add_model (gRPC) | Tried to add a model while learning is not running"
            )

        # Response
        return node_pb2.ResponseMessage()

    def handshake(self, request, _):
        """
        GRPC service. It is called when a node connects to another.
        """
        if self.round is not None:
            logging.info(
                f"({self.addr}) Cant connect to other nodes when learning is running."
            )
            return node_pb2.ResponseMessage(error="Cant connect: learning is running")
        else:
            return super().handshake(request, _)

    #########################
    #    Node Management    #
    #########################

    def connect(self, addr):
        """
        Connects a node to another. If learning is running, connections are not allowed.

        Args:
            addr (str): Address of the node to connect to.

        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        # Check if learning is running
        if self.round is not None:
            logging.info(
                f"({self.addr}) Cant connect to other nodes when learning is running."
            )
            return False
        # Connect
        return super().connect(addr)

    def stop(self):
        """
        Stops the node. If learning is running, the local learning process is interrupted.
        """
        # Interrupt learning
        if self.round is not None:
            self.__stop_learning()
        # Close node
        super().stop()

    def __start_reporter_thread(self):
        learning_thread = threading.Thread(
            target=self.__start_reporter
        )
        learning_thread.name = "reporter_thread-" + self.addr
        learning_thread.daemon = True
        learning_thread.start()

    def __start_reporter(self):
        while True:
            time.sleep(5)
            self.__report_status_to_controller()
            self.__report_resources()

    def __report_status_to_controller(self):
        """
        Report the status of the node to the controller.
        The configuration is the one that the node has at the memory.

        Returns:

        """
        # Set the URL for the POST request
        url = f'http://{self.config.participant["scenario_args"]["controller"]}/scenario/{self.config.participant["scenario_args"]["name"]}/node/update'

        # Send the POST request if the controller is available
        try:
            response = requests.post(url, data=json.dumps(self.config.participant),
                                     headers={'Content-Type': 'application/json'})
        except requests.exceptions.ConnectionError:
            logging.error(f'Error connecting to the controller at {url}')
            return

        # If endpoint is not available, log the error
        if response.status_code != 200:
            logging.error(f'Error received from controller: {response.status_code}')
            logging.error(response.text)

    def __report_resources(self):
        """
        Report node resources to the controller.

        Returns:

        """
        step = int((datetime.now() - datetime.strptime(self.config.participant["scenario_args"]["start_time"],
                                                       "%d/%m/%Y %H:%M:%S")).total_seconds())
        import sys
        import psutil
        # Gather CPU usage information
        cpu_percent = psutil.cpu_percent()
        # Gather CPU temperature information
        cpu_temp = 0
        try:
            if sys.platform == "linux":
                cpu_temp = psutil.sensors_temperatures()['coretemp'][0].current
        except Exception as e:
            pass
            # logging.error(f'Error getting CPU temperature: {e}')
        # Gather RAM usage information
        ram_percent = psutil.virtual_memory().percent
        # Gather disk usage information
        disk_percent = psutil.disk_usage("/").percent

        # Gather network usage information
        net_io_counters = psutil.net_io_counters()
        bytes_sent = net_io_counters.bytes_sent
        bytes_recv = net_io_counters.bytes_recv
        packets_sent = net_io_counters.packets_sent
        packets_recv = net_io_counters.packets_recv

        # Log uptime information
        uptime = psutil.boot_time()

        # Logging and reporting
        # logging.info(f'Resources: CPU {cpu_percent}%, CPU temp {cpu_temp}C, RAM {ram_percent}%, Disk {disk_percent}%')
        self.learner.logger.log_metrics(
            {"Resources/CPU_percent": cpu_percent, "Resources/CPU_temp": cpu_temp, "Resources/RAM_percent": ram_percent,
             "Resources/Disk_percent": disk_percent}, step=step)
        # logging.info(f'Resources: Bytes sent {bytes_sent}, Bytes recv {bytes_recv}, Packets sent {packets_sent}, Packets recv {packets_recv}')
        self.learner.logger.log_metrics({"Resources/Bytes_sent": bytes_sent, "Resources/Bytes_recv": bytes_recv,
                                         "Resources/Packets_sent": packets_sent,
                                         "Resources/Packets_recv": packets_recv}, step=step)
        # logging.info(f'Resources: Uptime {uptime}')
        self.learner.logger.log_metrics({"Resources/Uptime": uptime}, step=step)

        self.learner.logger.log_metrics({"Network/Connected": len(self.get_neighbors(only_direct=True))}, step=step)

        # Check if pynvml is available
        try:
            import pynvml
            pynvml.nvmlInit()
            devices = pynvml.nvmlDeviceGetCount()
            for i in range(devices):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_percent = gpu_mem.used / gpu_mem.total * 100
                # logging.info(f'Resources: GPU-{i} {gpu_percent}%, GPU temp {gpu_temp}C, GPU mem {gpu_mem_percent}%')
                self.learner.logger.log_metrics(
                    {f"Resources/GPU{i}_percent": gpu_percent, f"Resources/GPU{i}_temp": gpu_temp,
                     f"Resources/GPU{i}_mem_percent": gpu_mem_percent}, step=step)
        except ModuleNotFoundError:
            pass
            # logging.info(f'pynvml not found, skipping GPU usage')

    ##########################
    #    Learning Setters    #
    ##########################

    def set_data(self, data):
        """
        Set the data to be used in the learning process (by the learner).

        Args:
            data: Dataset to be used in the learning process.
        """
        self.learner.set_data(data)

    def set_model(self, model):
        """
        Set the model to be used in the learning process (by the learner).

        Args:
            model: Model to be used in the learning process.
        """
        self.learner.set_model(model)

    ###############################################
    #         Network Learning Management         #
    ###############################################

    def set_start_learning(self, rounds=1, epochs=1):
        """
        Start the learning process in the entire network.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        self.assert_running(True)

        if self.round is None:
            # Broadcast start Learning
            logging.info(f"({self.addr}) Broadcasting start learning...")
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(
                    LearningNodeMessages.START_LEARNING, [rounds, epochs]
                )
            )
            # Set model initialized
            self.__model_initialized_lock.release()
            # Broadcast initialize model
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(LearningNodeMessages.MODEL_INITIALIZED)
            )
            # Learning Thread
            self.__start_learning_thread(rounds, epochs)
        else:
            logging.info(f"({self.addr}) Learning already started")

    def set_stop_learning(self):
        """
        Stop the learning process in the entire network.
        """
        if self.round is not None:
            # send stop msg
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(LearningNodeMessages.STOP_LEARNING)
            )
            # stop learning
            self.__stop_learning()
        else:
            logging.info(f"({self.addr}) Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning_thread(self, rounds, epochs):
        learning_thread = threading.Thread(
            target=self.__start_learning, args=(rounds, epochs)
        )
        learning_thread.name = "learning_thread-" + self.addr
        learning_thread.daemon = True
        learning_thread.start()

    def __start_learning(self, rounds, epochs):
        self.__start_thread_lock.acquire()  # Used to avoid create duplicated training threads
        if self.round is None:
            self.round = 0
            self.totalrounds = rounds
            self.__start_thread_lock.release()
            begin = time.time()

            # Wait and gossip model initialization
            logging.info(f"({self.addr}) Waiting initialization.")
            self.__model_initialized_lock.acquire()
            logging.info(f"({self.addr}) Gossiping model initialization.")
            self.__gossip_model_difusion(initialization=True)

            # Wait to guarantee new connection heartbeats convergence
            wait_time = self.config.participant["WAIT_HEARTBEATS_CONVERGENCE"] - (time.time() - begin)
            if wait_time > 0:
                time.sleep(wait_time)

            # Train
            self.learner.set_epochs(epochs)
            self.learner.create_trainer()
            self.__train_step()
        else:
            self.__start_thread_lock.release()

    def __stop_learning(self):
        logging.info(f"({self.addr}) Stopping learning")
        # Rounds
        self.round = None
        self.totalrounds = None
        self.learner.interrupt_fit()
        # Aggregator
        self.aggregator.clear()
        # Try to free wait locks
        try:
            self.__wait_votes_ready_lock.release()
        except:
            pass

    #######################
    #    Training Steps    #
    #######################

    def __wait_aggregated_model(self):
        params = self.aggregator.wait_and_get_aggregation()

        # Set parameters and communate it to the training process
        if params is not None:
            logging.info(
                f"({self.addr}) __wait_aggregated_model | Aggregation done for round {self.round}, including parameters in local model.")
            self.learner.set_parameters(params)
            logging.debug(
                f"({self.addr}) Broadcast aggregation done for round {self.round}"
            )
            # Share that aggregation is done
            logging.info(
                f"({self.addr}) __wait_aggregated_model | Broadcasting aggregation done for round {self.round}")
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(
                    LearningNodeMessages.MODELS_READY, [self.round]
                )
            )
        else:
            logging.error(f"({self.addr}) Aggregation finished with no parameters")
            self.stop()

    def __train_step(self):
        # Set train set
        if self.round is not None:
            # self.__train_set = self.__vote_train_set() # TODO: Remove vote train set
            self.__train_set = self.get_neighbors(only_direct=False)
            self.__train_set = self.__validate_train_set(self.__train_set)
            if self.addr not in self.__train_set:
                self.__train_set.append(self.addr)
            logging.info(
                f"{self.addr} Train set: {self.__train_set}"
            )
            # Logging neighbors (indicate the direct neighbors and undirected neighbors)
            logging.info(
                f"{self.addr} All neighbors: {self.get_neighbors()} | Direct neighbors: {self.get_neighbors(only_direct=True)}"
            )

        # Determine if node is in the train set
        if self.addr in self.__train_set and (self.config.participant["device_args"]["role"] == Role.AGGREGATOR or
                                              self.config.participant["device_args"]["role"] == Role.SERVER):
            logging.info("[NODE.__train_step] Role.AGGREGATOR/Role.SERVER process...")
            if self.round is not None:
                # Set Models To Aggregate
                self.aggregator.set_nodes_to_aggregate(self.__train_set)

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()

            # Aggregate Model
            if self.round is not None:
                models_added = self.aggregator.add_model(
                    self.learner.get_parameters(),
                    [self.addr],
                    self.learner.get_num_samples()[0],
                    source=self.addr,
                    round=self.round
                )
                # send model added msg ---->> redundant (a node always owns its model)
                self._neighbors.broadcast_msg(
                    self._neighbors.build_msg(
                        LearningNodeMessages.MODELS_AGGREGATED, models_added
                    )
                )
                if self.config.participant["device_args"]["role"] == Role.SERVER:
                    self.__gossip_model_difusion()
                else:
                    self.__gossip_model_aggregation()

        elif self.config.participant["device_args"]["role"] == Role.TRAINER:
            logging.info("[NODE.__train_step] Role.TRAINER process...")
            if self.round is not None:
                # Set Models To Aggregate
                self.aggregator.set_nodes_to_aggregate(self.__train_set)

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()

            # Aggregate Model
            if self.round is not None:
                models_added = self.aggregator.add_model(
                    self.learner.get_parameters(),
                    [self.addr],
                    self.learner.get_num_samples()[0],
                    source=self.addr,
                    round=self.round
                )
                # send model added msg ---->> redundant (a node always owns its model)
                self._neighbors.broadcast_msg(
                    self._neighbors.build_msg(
                        LearningNodeMessages.MODELS_AGGREGATED, models_added
                    )
                )
                self.__gossip_model_aggregation()
                # self.__wait_aggregated_model()
                self.aggregator.set_waiting_aggregated_model(self.__train_set)
                time.sleep(5)

        elif self.config.participant["device_args"]["role"] == Role.IDLE:
            # self.__wait_aggregated_model()
            self.aggregator.set_waiting_aggregated_model(self.__train_set)

        else:
            logging.warning("[NODE.__train_step] Role not implemented yet")

        # Gossip aggregated model
        if self.round is not None:
            logging.info(f"({self.addr}) Waiting aggregation and gossiping model (difusion).")
            self.__wait_aggregated_model()
            self.__gossip_model_difusion()

        # Finish round
        if self.round is not None:
            self.__on_round_finished()

    ################
    #    Voting    #
    ################

    def __vote_train_set(self):
        # Vote (at least itself)
        candidates = list(self.get_neighbors(only_direct=False))
        if self.addr not in candidates:
            candidates.append(self.addr)
        logging.debug(f"({self.addr}) {len(candidates)} candidates to train set")

        # Send vote
        samples = min(self.config.participant["TRAIN_SET_SIZE"], len(candidates))
        nodes_voted = random.sample(candidates, samples)
        weights = [
            math.floor(random.randint(0, 1000) / (i + 1)) for i in range(samples)
        ]
        votes = list(zip(nodes_voted, weights))

        # Adding votes
        self.__train_set_votes_lock.acquire()
        self.__train_set_votes[self.addr] = dict(votes)
        self.__train_set_votes_lock.release()

        # Send and wait for votes
        logging.info(f"({self.addr}) Sending train set vote.")
        logging.debug(f"({self.addr}) Self Vote: {votes}")
        self._neighbors.broadcast_msg(
            self._neighbors.build_msg(
                LearningNodeMessages.VOTE_TRAIN_SET,
                list(sum(votes, tuple())),
                round=self.round,
            )
        )
        logging.debug(f"({self.addr}) Waiting other node votes.")

        # Get time
        count = 0
        begin = time.time()

        while True:
            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info(f"({self.addr}) Stopping on_round_finished process.")
                return []

            # Update time counters (timeout)
            count = count + (time.time() - begin)
            begin = time.time()
            timeout = count > self.config.participant["VOTE_TIMEOUT"]

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
                        f"({self.addr}) Timeout for vote aggregation. Missing votes from {set(candidates) - set(nc_votes.keys())}"
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
                top = min(len(results), self.config.participant["TRAIN_SET_SIZE"])
                results = results[0:top]
                results = {k: v for k, v in results}
                votes = list(results.keys())

                # Clear votes
                self.__train_set_votes = {}
                logging.info(f"({self.addr}) Computed {len(nc_votes)} votes.")
                return votes

            # Wait for votes or refresh every 2 seconds
            self.__wait_votes_ready_lock.acquire(timeout=2)

    def __validate_train_set(self, train_set):
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        for tsn in train_set:
            if tsn not in self.get_neighbors(only_direct=False):
                if tsn != self.addr:
                    train_set.remove(tsn)
        return train_set

    ############################
    #    Train and Evaluate    #
    ############################

    def __train(self):
        logging.info(f"({self.addr}) Training...")
        self.learner.fit()
        logging.info(f"({self.addr}) Finished training.")

    def __evaluate(self):
        logging.info(f"({self.addr}) Evaluating...")
        results = self.learner.evaluate()
        logging.info(f"({self.addr}) Finished evaluating.")
        # Removed because it is not necessary to send metrics between nodes
        if results is not None:
            logging.info(
                f"({self.addr}) Evaluated. Losss: {results[0]}, Metric: {results[1]}."
            )
            # Send metrics
            logging.info(f"({self.addr}) Broadcasting metrics.")
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(
                    LearningNodeMessages.METRICS,
                    [results[0], results[1]],
                    round=self.round,
                )
            )

    ######################
    #    Round finish    #
    ######################

    def __on_round_finished(self):
        # Set Next Round
        # implement a lock to avoid concurrent access to round
        self.finish_round_lock.acquire()
        logging.info(
            f"({self.addr}) Round {self.round} of {self.totalrounds} finished."
        )
        self.aggregator.clear()
        self.learner.finalize_round()  # check to see if this could look better
        self.round = self.round + 1
        self.config.participant["federation_args"]["round"] = self.round  # Set current round in config (it is sent to the controller)
        logging.info(f"({self.addr}) Round {self.round} of {self.totalrounds} started.")
        self.aggregator.set_round(self.round)
        # Clear node aggregation
        self.__models_aggregated = {}
        self.finish_round_lock.release()

        # Next Step or Finish
        if self.round < self.totalrounds:
            self.__train_step()
        else:
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            self.round = None
            self.totalrounds = None
            self.__model_initialized_lock.acquire()
            logging.info(f"({self.addr}) Training finished!!.")

    #########################
    #    Model Gossiping    #
    #########################

    def reputation_calculation(self, aggregated_models_weights):
        # Compare the model parameters to identify malicious nodes, and then broadcast to the rest of the topology
        # Functionality not implemented yet (ROADMAP 1.0)
        # ...
        cossim_threshold = 0.5
        loss_threshold = 0.5

        current_models = {}
        for subnodes in aggregated_models_weights.keys():
            sublist = subnodes.split()
            submodel = aggregated_models_weights[subnodes][0]
            for node in sublist:
                current_models[node] = submodel
        # logging.info(f'reputation_calculation current_models {current_models}')
        malicious_nodes = []
        reputation_score = {}
        local_model = self.learner.get_parameters()
        untrusted_nodes = list(current_models.keys())
        # logging.info(f'reputation_calculation untrusted_nodes {untrusted_nodes}')
        selfName = self.get_name()
        # selfName = ''
        logging.info(f'reputation_calculation untrusted_nodes at round {self.learner.get_round()}: {untrusted_nodes}')
        for untrusted_node in untrusted_nodes:
            logging.info(f'reputation_calculation untrusted_node at round {self.learner.get_round()}: {untrusted_node}')
            logging.info(f'reputation_calculation selfName at round {self.learner.get_round()}: {selfName}')
            if untrusted_node != selfName:
                untrusted_model = current_models[untrusted_node]
                cossim = cosine_similarity(local_model, untrusted_model)
                logging.info(
                    f'reputation_calculation cossim at round {self.learner.get_round()}: {untrusted_node}: {cossim}')
                self.learner.logger.log_metrics({f"Reputation/cossim_{untrusted_node}": cossim},
                                                step=self.learner.get_round())

                avg_loss = self.learner.validate_neighbour_model(untrusted_model)
                logging.info(
                    f'reputation_calculation avg_loss at round {self.learner.get_round()} {untrusted_node}: {avg_loss}')
                self.learner.logger.log_metrics({f"Reputation/avg_loss_{untrusted_node}": avg_loss},
                                                step=self.learner.get_round())
                reputation_score[untrusted_node] = (cossim, avg_loss)

                if cossim < cossim_threshold or avg_loss > loss_threshold:
                    malicious_nodes.append(untrusted_node)
                else:
                    self.__trusted_nei.append(untrusted_node)

        return malicious_nodes, reputation_score

    def send_reputation(self, malicious_nodes):
        logging.info(f"({self.addr}) Broadcasting reputation message...")
        self._neighbors.broadcast_msg(
            self._neighbors.build_msg(
                LearningNodeMessages.REPUTATION, malicious_nodes
            )
        )

    def get_aggregated_models(self, node):
        """
        Get the models that have been aggregated by a given node in the actual round.

        Args:
            node (str): Node to get the aggregated models from.
        """
        try:
            if self.with_reputation:
                malicious_nodes = []
                # logging.info(f"({self.addr}) Stored models: {self.aggregator.get_aggregated_models_weights()}")
                if self.learner.get_round() > 2:
                    malicious_nodes, _ = self.reputation_calculation(self.aggregator.get_aggregated_models_weights())
                    logging.info(
                        f"({self.addr}) Malicious nodes  at round {self.learner.get_round()}: {malicious_nodes}, excluding them from the aggregation...")
                    if len(malicious_nodes) > 0:
                        self.send_reputation(malicious_nodes)
                        # disrupt the connection with malicious nodes
                        if self.is_dynamic_topology:
                            self.__disrupt_connection_using_reputation(malicious_nodes)

                        if self.is_dynamic_aggregation and self.aggregator != self.target_aggregation:
                            self.__dynamic_aggergator(self.aggregator.get_aggregated_models_weights(), malicious_nodes)

                # Exclude malicious nodes from the aggregation
                # Introduce the malicious nodes in the list of aggregated models. Then remove the duplicates
                models_aggregated = self.__models_aggregated[node]
                models_aggregated = list(set(list(models_aggregated) + malicious_nodes))
                return models_aggregated
            else:
                return self.__models_aggregated[node]
        except KeyError:
            return []

    def __gossip_model_aggregation(self):
        # Anonymous functions
        logging.info(f"({self.addr}) __gossip_model_aggregation")
        candidate_condition = lambda node: (
                (node not in self.aggregator.get_aggregated_models())
                and (node in self.__train_set)
        )
        status_function = lambda node: (
            node,
            self.get_aggregated_models(node),
        )
        model_function = lambda node: self.aggregator.get_partial_aggregation(
            self.get_aggregated_models(node)
        )

        # Gossip
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model_difusion(self, initialization=False):
        # Wait a model (init or aggregated)
        logging.info(f"({self.addr}) __gossip_model_difusion")
        if initialization:
            logging.info(f"({self.addr}) __gossip_model_difusion | Waiting model initialization.")
            candidate_condition = lambda node: node not in self.__nei_status.keys()
        else:
            logging.info(f"({self.addr}) __gossip_model_difusion | Waiting model aggregation.")
            candidate_condition = lambda node: self.__nei_status[node] < self.round

        # Status fn
        status_function = lambda nc: nc
        # Model fn -> At diffusion, contributors are not relevant
        model_function = lambda _: (
            self.learner.get_parameters(),
            self.aggregator.get_aggregated_models(),
            1,
        )

        # Gossip
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model(
            self,
            candidate_condition,
            status_function,
            model_function
    ):
        period = self.config.participant["GOSSIP_MODELS_PERIOD"]
        # Initialize list with status of nodes in the last X iterations
        last_x_status = []
        j = 0

        while True:
            # Get time to calculate frequency
            t = time.time()

            # If the training has been interrupted, stop waiting
            if self.round is None:
                logging.info(f"({self.addr}) Gossip | Stopping model gossip process.")
                return

            # Get nodes which need models
            neis = [n for n in self.get_neighbors() if candidate_condition(n)]
            logging.info(
                f"({self.addr} Gossip | {neis} need models --> (node not in self.aggregator.get_aggregated_models()) and (node in self.__train_set)")

            # Determine end of gossip
            if not neis:
                logging.info(f"({self.addr}) Gossip| Gossip finished. No more nodes need models.")
                return

            # Save state of neighbors. If nodes are not responding gossip will stop
            if len(last_x_status) != self.config.participant["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]:
                last_x_status.append([status_function(n) for n in neis])
            else:
                last_x_status[j] = str([status_function(n) for n in neis])
                j = (j + 1) % self.config.participant["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]

                # Check if las messages are the same
                for i in range(len(last_x_status) - 1):
                    if last_x_status[i] != last_x_status[i + 1]:
                        break
                    logging.info(
                        f"({self.addr}) Gossip | Gossiping exited for {self.config.participant['GOSSIP_EXIT_ON_X_EQUAL_ROUNDS']} equal rounds."
                    )
                    return

            # Select a random subset of neighbors
            samples = min(self.config.participant["GOSSIP_MODELS_PER_ROUND"], len(neis))
            neis = random.sample(neis, samples)
            logging.info(f"({self.addr}) Gossip | Gossiping models to {neis}")

            # Generate and Send Model Partial Aggregations (model, node_contributors)
            for nei in neis:
                model, contributors, weight = model_function(nei)

                # Send Partial Aggregation
                if model is not None:
                    logging.info(
                        f"({self.addr}) Gossip | Gossiping model (partial aggregation) to {nei} with contributors: {contributors} and weight: {weight}")
                    encoded_model = self.learner.encode_parameters(params=model)
                    self._neighbors.send_model(
                        nei, self.round, encoded_model, contributors, weight
                    )

            # Sleep to allow periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)


class MaliciousNode(Node):

    def __init__(self,
                 idx,
                 experiment_name,
                 model,
                 data,
                 hostdemo=None,
                 host="127.0.0.1",
                 port=None,
                 config=Config,
                 learner=LightningLearner,
                 encrypt=False,
                 model_poisoning=False,
                 poisoned_ratio=0,
                 noise_type='gaussian'):
        """Node that instead of training, performs an attack on the weights during the aggregation.
        """
        super().__init__(idx, experiment_name, model, data, hostdemo, host, port, config, learner, encrypt)

        # Create attack object
        self.attack = create_attack(config.participant["adversarial_args"]["attacks"])
        self.fit_time = 0.0
        # Time it would wait additionally to the normal training time
        self.extra_time = 0.0
        self.aggregator = create_malicious_aggregator(self.aggregator, self.attack)

    def _Node__train(self):  # Required to override Node.__train method
        if self.learner.get_round() == 0:
            t0 = time.time()
            logging.info(f"({self.addr}) Training...")
            self.learner.fit()
            self.fit_time = time.time() - t0 + self.extra_time
        else:
            logging.info(f"({self.addr}) Waiting {self.fit_time} seconds maliciously...")
            time.sleep(max(self.fit_time, 0.0))
