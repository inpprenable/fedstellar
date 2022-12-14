import glob
import hashlib
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from fedstellar.config.config import Config
from fedstellar.config.mender import Mender
from fedstellar.utils.topologymanager import TopologyManager
from fedstellar.webserver.app import run_webserver

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Setup controller logger
log_console_format = "\x1b[0;35m[%(levelname)s] - %(asctime)s - Controller -\x1b[0m %(message)s"
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_console_format))
logging.basicConfig(level=logging.DEBUG,
                    handlers=[
                        console_handler,
                    ])


# Detect ctrl+c and run killports
def signal_handler(sig, frame):
    logging.info('You pressed Ctrl+C!')
    Controller.killports()
    os.system("""osascript -e 'tell application "Terminal" to quit'""") if sys.platform == "darwin" else None
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Controller:
    """
    Controller class that manages the nodes
    """

    def __init__(self, args):
        self.experiment_name = f'fedstellar_{args.federation}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
        self.federation = args.federation
        self.topology = args.topology
        self.webserver = args.webserver
        self.simulation = args.simulation
        self.config_dir = args.config
        self.log_dir = args.logs
        self.env_path = args.env

        self.config = None
        self.topologymanager = None
        self.n_nodes = 0
        self.mender = None if self.simulation else Mender()

    def start(self):
        """
        Start the controller
        """
        self.inicialization()

        logging.info("Generation logs directory for experiment: {}".format(self.experiment_name))
        os.makedirs(os.path.join(self.log_dir, self.experiment_name), exist_ok=True)

        self.config = Config(entity="controller")
        # Get participants configurations
        participant_files = glob.glob('{}/participant_*.json'.format(self.config_dir))
        participant_files.sort()
        if len(participant_files) == 0:
            raise ValueError("No participant files found in config folder")

        self.config.set_participants_config(participant_files)
        self.n_nodes = len(participant_files)
        logging.info("Number of nodes: {}".format(self.n_nodes))

        # self.topologymanager = self.create_topology()
        self.topologymanager = self.create_topology(matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        # Update participants configuration
        is_start_node = False
        for i in range(self.n_nodes):
            with open(f'{self.config_dir}/participant_' + str(i) + '.json') as f:
                participant_config = json.load(f)
            participant_config['scenario_args']["federation"] = self.federation
            participant_config['scenario_args']['n_nodes'] = self.n_nodes
            participant_config['network_args']['neighbors'] = self.topologymanager.get_neighbors_string(i)
            participant_config['scenario_args']['name'] = self.experiment_name
            participant_config['device_args']['idx'] = i
            participant_config['device_args']['uid'] = hashlib.sha1((str(participant_config["network_args"]["ip"]) + str(participant_config["network_args"]["port"])).encode()).hexdigest()
            participant_config['tracking_args']['log_dir'] = self.log_dir
            participant_config['tracking_args']['config_dir'] = self.config_dir
            if participant_config["device_args"]["start"]:
                if not is_start_node:
                    is_start_node = True
                else:
                    raise ValueError("Only one node can be start node")
            with open('config/participant_' + str(i) + '.json', 'w') as f:
                json.dump(participant_config, f, sort_keys=False, indent=2)
        if not is_start_node:
            raise ValueError("No start node found")
        self.config.set_participants_config(participant_files)

        # Add role to the topology (visualization purposes)
        self.topologymanager.update_nodes(self.config.participants)
        self.topologymanager.draw_graph(path=f"{self.log_dir}/{self.experiment_name}/topology.png", plot=True)

        topology_json_path = "{}/topology.json".format(self.config_dir)
        self.topologymanager.update_topology_3d_json(participants=self.config.participants, path=topology_json_path)

        webserver = True  # TODO: change it
        if webserver:
            logging.info("Starting webserver")
            server_process = multiprocessing.Process(target=run_webserver)  # Also, webserver can be started manually
            server_process.start()

        # while True:
        #    time.sleep(1)

        if self.mender:
            logging.info("[Mender.module] Mender module initialized")
            time.sleep(2)
            mender = Mender()
            logging.info("[Mender.module] Getting token from Mender server: {}".format(os.getenv("MENDER_SERVER")))
            mender.renew_token()
            time.sleep(2)
            logging.info("[Mender.module] Getting devices from {} with group Cluster_Thun".format(os.getenv("MENDER_SERVER")))
            time.sleep(2)
            devices = mender.get_devices_by_group("Cluster_Thun")
            logging.info("[Mender.module] Getting a pool of devices: 5 devices")
            # devices = devices[:5]
            for i in self.config.participants:
                logging.info("[Mender.module] Device {} | IP: {}".format(i['device_args']['idx'], i['network_args']['ipdemo']))
                logging.info("[Mender.module] \tCreating artifacts...")
                logging.info("[Mender.module] \tSending Fedstellar framework...")
                # mender.deploy_artifact_device("my-update-2.0.mender", i['device_args']['idx'])
                logging.info("[Mender.module] \tSending configuration...")
                time.sleep(5)

        self.start_nodes()

        logging.info('Press Ctrl+C for exit')
        while True:
            time.sleep(1)

    def inicialization(self):
        # First, kill all the ports related to previous executions
        self.killports()

        banner = """
                    ______       _     _       _ _            
                    |  ___|     | |   | |     | | |           
                    | |_ ___  __| |___| |_ ___| | | __ _ _ __ 
                    |  _/ _ \/ _` / __| __/ _ \ | |/ _` | '__|
                    | ||  __/ (_| \__ \ ||  __/ | | (_| | |   
                    \_| \___|\__,_|___/\__\___|_|_|\__,_|_|   
                A Framework for Decentralized Federated Learning 
               Enrique Tomás Martínez Beltrán (enriquetomas@um.es)
            """
        print("\x1b[0;36m" + banner + "\x1b[0m")

        # Load the environment variables
        load_dotenv(self.env_path)

        # Get some info about the backend
        # collect_env()

        from netifaces import AF_INET
        import netifaces as ni
        ip_address = ni.ifaddresses('en0')[AF_INET][0]['addr']
        import ipaddress
        network = ipaddress.IPv4Network(f"{ip_address}/24", strict=False)

        logging.info("Controller network: {}".format(network))
        logging.info("Controller IP address: {}".format(ip_address))
        logging.info("Federated architecture: {}".format(self.federation))

    @staticmethod
    def killports(term="python"):
        # kill all the ports related to python processes
        time.sleep(1)
        command = '''kill -9 $(lsof -i @localhost:1024-65545 | grep ''' + term + ''' | awk '{print $2}') > /dev/null 2>&1'''
        os.system(command)

    def create_topology(self, matrix=None):
        import numpy as np
        if matrix is not None:
            topologymanager = TopologyManager(topology=np.array(matrix), experiment_name=self.experiment_name, log_dir=self.log_dir, n_nodes=self.n_nodes, b_symmetric=True, undirected_neighbor_num=self.n_nodes - 1)
        elif self.topology == "fully":
            # Create a fully connected network
            topologymanager = TopologyManager(experiment_name=self.experiment_name, log_dir=self.log_dir, n_nodes=self.n_nodes, b_symmetric=True, undirected_neighbor_num=self.n_nodes - 1)
            topologymanager.generate_topology()
        elif self.topology == "ring":
            # Create a partially connected network (ring-structured network)
            topologymanager = TopologyManager(experiment_name=self.experiment_name, log_dir=self.log_dir, n_nodes=self.n_nodes, b_symmetric=True)
            topologymanager.generate_ring_topology(increase_convergence=True)
        elif self.topology == "random":
            # Create network topology using topology manager (random)
            topologymanager = TopologyManager(experiment_name=self.experiment_name, log_dir=self.log_dir, n_nodes=self.n_nodes, b_symmetric=True,
                                              undirected_neighbor_num=3)
            topologymanager.generate_topology()
        elif self.topology == "star" and self.federation == "CFL":
            # Create a centralized network
            topologymanager = TopologyManager(experiment_name=self.experiment_name, log_dir=self.log_dir, n_nodes=self.n_nodes, b_symmetric=True)
            topologymanager.generate_server_topology()
        else:
            raise ValueError("Unknown topology type: {}".format(self.topology))

        # topology = topologymanager.get_topology()
        # logging.info(topology)

        # Also, it is possible to use a custom topology using adjacency matrix
        # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, topology=[[0, 1, 1, 1], [1, 0, 1, 1]. [1, 1, 0, 1], [1, 1, 1, 0]])

        # Assign nodes to topology
        nodes_ip_port = []
        for i, node in enumerate(self.config.participants):
            nodes_ip_port.append((node['network_args']['ip'], node['network_args']['port'], "undefined", node['network_args']['ipdemo']))

        topologymanager.add_nodes(nodes_ip_port)
        return topologymanager

    def start_nodes(self):
        # Change python path to the current environment (controller and participants)
        python_path = '/Users/enrique/miniforge3/envs/phd/bin/python'

        for idx in range(0, self.n_nodes):
            logging.info("Starting node {} with configuration {}".format(idx, self.config.participants[idx]))
            command = 'cd /Users/enrique/Documents/PhD/fedstellar/fedstellar' + '; ' + python_path + ' -u node_start.py ' \
                      + str(self.config.participants_path[idx]) + ' 2>&1'
            if sys.platform == "darwin":
                os.system("""osascript -e 'tell application "Terminal" to activate' -e 'tell application "Terminal" to do script "{}"'""".format(command))
            else:
                os.system(command)