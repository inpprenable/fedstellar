import argparse
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

# Add contents root_dir directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import fedstellar
from fedstellar.config.config import Config
from fedstellar.utils.topologymanager import TopologyManager
from fedstellar.config.mender import Mender

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

argparser = argparse.ArgumentParser(description='Controller of Fedstellar framework', add_help=False)

argparser.add_argument('-n', '--name', dest='name',
                       default="{}".format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S")),
                       help='Experiment name')
argparser.add_argument('-f', '--federation', dest='federation', default="SDFL",
                       help='Federation architecture: CFL, DFL, or SDFL (default: DFL)')
argparser.add_argument('-t', '--topology', dest='topology', default="fully",
                          help='Topology: fully, ring, random, or star (default: fully)')
argparser.add_argument('-s', '--simulation', action='store_false', dest='simulation', help='Run simulation')
argparser.add_argument('-v', '--version', action='version',
                       version='%(prog)s ' + fedstellar.__version__, help="Show version")
argparser.add_argument('-a', '--about', action='version',
                       version='Created by Enrique Tomás Martínez Beltrán',
                       help="Show author")
argparser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                       help='Show help')

args = argparser.parse_args()

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
    killports()
    os.system("""osascript -e 'tell application "Terminal" to quit'""") if sys.platform == "darwin" else None
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def killports(term="python"):
    # kill all the ports related to python processes
    time.sleep(1)
    command = '''kill -9 $(lsof -i @localhost:1024-65545 | grep ''' + term + ''' | awk '{print $2}') > /dev/null 2>&1'''
    os.system(command)


def create_topology(config, experiment_name, n_nodes):
    if args.topology == "fully":
        # Create a fully connected network
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True, undirected_neighbor_num=n_nodes - 1)
        topologymanager.generate_topology()
    elif args.topology == "ring":
        # Create a partially connected network (ring-structured network)
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True)
        topologymanager.generate_ring_topology(increase_convergence=True)
    elif args.topology == "random":
        # Create network topology using topology manager (random)
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True,
                                          undirected_neighbor_num=3)
        topologymanager.generate_topology()
    elif args.topology == "star" and args.federation == "CFL":
        # Create a centralized network
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True)
        topologymanager.generate_server_topology()
    else:
        raise ValueError("Unknown topology type: {}".format(args.topology))

    # topology = topologymanager.get_topology()
    # logging.info(topology)

    # Also, it is possible to use a custom topology using adjacency matrix
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, topology=[[0, 1, 1, 1], [1, 0, 1, 1]. [1, 1, 0, 1], [1, 1, 1, 0]])

    # Assign nodes to topology
    nodes_ip_port = []
    for i, node in enumerate(config.participants):
        nodes_ip_port.append((node['network_args']['ip'], node['network_args']['port'], "undefined", node['network_args']['ipdemo']))

    topologymanager.add_nodes(nodes_ip_port)
    return topologymanager


def main():
    # First, kill all the ports related to previous executions
    killports()

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

    experiment_name = f"fedstellar_{args.federation}_{args.name}"

    # Load the environment variables
    envpath = os.path.join(os.path.dirname(__file__), '../fedstellar/.env')
    envpath = os.path.abspath(envpath)
    load_dotenv(envpath)

    # Get some info about the backend
    # collect_env()

    from netifaces import AF_INET
    import netifaces as ni
    ip_address = ni.ifaddresses('en0')[AF_INET][0]['addr']
    import ipaddress
    network = ipaddress.IPv4Network(f"{ip_address}/24", strict=False)

    logging.info("Controller network: {}".format(network))
    logging.info("Controller IP address: {}".format(ip_address))
    logging.info("Federated architecture: {}".format(args.federation))

    config = Config(entity="controller")

    participant_files = glob.glob(os.path.join(os.path.dirname(__file__), 'config/participant_*.json'))
    participant_files.sort()
    if len(participant_files) == 0:
        raise ValueError("No participant files found in config folder")

    config.set_participants_config(participant_files)

    n_nodes = len(participant_files)
    logging.info("Number of nodes: {}".format(n_nodes))

    # input("Topology and participant configuration files generated. Check and press any key to continue...\n")

    if not args.simulation:
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

    logging.info("Generation logs directory for experiment: {}".format(experiment_name))
    os.makedirs(os.path.join(os.path.dirname(__file__), 'logs/' + experiment_name), exist_ok=True)

    logging.info("Generating topology configuration file...")
    topologymanager = create_topology(config, experiment_name, n_nodes)

    # Update participants configuration
    is_start_node = False
    for i in range(n_nodes):
        with open('config/participant_' + str(i) + '.json') as f:
            participant_config = json.load(f)
        participant_config['network_args']['neighbors'] = topologymanager.get_neighbors_string(i)
        participant_config['scenario_args']['name'] = experiment_name
        participant_config['device_args']['idx'] = i
        participant_config['device_args']['uid'] = hashlib.sha1((str(participant_config["network_args"]["ip"]) + str(participant_config["network_args"]["port"])).encode()).hexdigest()
        if participant_config["device_args"]["start"]:
            if not is_start_node:
                is_start_node = True
            else:
                raise ValueError("Only one node can be start node")
        with open('config/participant_' + str(i) + '.json', 'w') as f:
            json.dump(participant_config, f, sort_keys=False, indent=2)
    if not is_start_node:
        raise ValueError("No start node found")
    config.set_participants_config(participant_files)

    if not args.simulation:
        for i in config.participants:
            logging.info("[Mender.module] Device {} | IP: {}".format(i['device_args']['idx'], i['network_args']['ipdemo']))
            logging.info("[Mender.module] \tCreating artifacts...")
            logging.info("[Mender.module] \tSending Fedstellar framework...")
            # mender.deploy_artifact_device("my-update-2.0.mender", i['device_args']['idx'])
            logging.info("[Mender.module] \tSending configuration...")
            time.sleep(5)

    # Add role to the topology (visualization purposes)
    topologymanager.update_nodes(config.participants)
    topologymanager.draw_graph(plot=True)

    json_path = "{}/config/topology.json".format(sys.path[0])
    topologymanager.update_topology_3d_json(participants=config.participants, path=json_path)

    webserver = True  # TODO: change it
    if webserver:
        from fedstellar.webserver.app import run_webserver
        logging.info("Starting webserver")
        server_process = multiprocessing.Process(target=run_webserver)  # Also, webserver can be started manually
        server_process.start()

    # while True:
    #    time.sleep(1)

    # Change python path to the current environment (controller and participants)
    python_path = '/Users/enrique/miniforge3/envs/phd/bin/python'

    for idx in range(1, n_nodes):
        command = 'cd /Users/enrique/Documents/PhD/fedstellar/examples' + '; ' + python_path + ' -u node_start.py ' \
                  + str(idx) + ' 2>&1'
        if sys.platform == "darwin":
            os.system("""osascript -e 'tell application "Terminal" to activate' -e 'tell application "Terminal" to do script "{}"'""".format(command))
        else:
            os.system(command)

    start_node = True
    command = 'cd /Users/enrique/Documents/PhD/fedstellar/examples' + '; ' + python_path + ' -u node_start.py ' \
              + str(0) + ' 2>&1'
    if sys.platform == "darwin":
        os.system("""osascript -e 'tell application "Terminal" to activate' -e 'tell application "Terminal" to do script "{}"'""".format(command))
    else:
        os.system(command)

    logging.info('Press Ctrl+C for exit')
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
