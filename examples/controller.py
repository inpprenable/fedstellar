import argparse
import hashlib
import logging
import multiprocessing
import os
import pickle
import shutil
import signal
import sys
import time
from datetime import datetime
import yaml

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
argparser.add_argument('-f', '--federation', dest='federation', default="DFL",
                       help='Federation architecture: CFL, DFL, or SDFL (default: DFL)')
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
    if config.topology['type'] == "fully":
        # Create a fully connected network
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True, undirected_neighbor_num=n_nodes - 1)
        topologymanager.generate_topology()
    elif config.topology['type'] == "ring":
        # Create a partially connected network (ring-structured network)
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True)
        topologymanager.generate_ring_topology(increase_convergence=True)
    elif config.topology['type'] == "random":
        # Create network topology using topology manager (random)
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True,
                                          undirected_neighbor_num=3)
        topologymanager.generate_topology()
    elif config.topology['type'] == "star" and args.federation == "CFL":
        # Create a centralized network
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True)
        topologymanager.generate_server_topology()
    else:
        raise ValueError("Unknown topology type: {}".format(config.topology['type']))

    # topology = topologymanager.get_topology()
    # logging.info(topology)

    # Also, it is possible to use a custom topology using adjacency matrix
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, topology=[[0, 1, 1, 1], [1, 0, 1, 1]. [1, 1, 0, 1], [1, 1, 1, 0]])

    # Assign nodes to topology
    nodes_ip_port = []
    for i, node in enumerate(config.topology['nodes']):
        nodes_ip_port.append((node['ip'], node['port'], "undefined", node['ipdemo']))

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

    config = Config(entity="controller")

    # Load the topology configuration
    topology_path = "config/topology.json"
    if not os.path.exists(topology_path):
        shutil.copyfile(os.path.join(os.path.dirname(__file__), '../fedstellar/config/topology.json.example'), topology_path)

    config.set_topology_config('config/topology.json')

    # Generate a participant configuration file for each node in the topology
    for i, node in enumerate(config.topology['nodes']):
        if not os.path.exists('config/participant_' + str(i) + '.yaml'):
            shutil.copyfile(os.path.join(os.path.dirname(__file__), '../fedstellar/config/participant.yaml.example'), 'config/participant_' + str(i) + '.yaml')

    input("Topology and participant configuration files generated. Check and press any key to continue...\n")

    n_nodes = len(config.topology['nodes'])

    logging.info("Controller network: {}".format(network))
    logging.info("Controller IP address: {}".format(ip_address))
    logging.info("Federated architecture: {}".format(args.federation))

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

    logging.info("Generating topology configuration file\n{}".format(config.get_topology_config()))
    topologymanager = create_topology(config, experiment_name, n_nodes)

    # Update participants configuration
    is_start_node = False
    for i in range(n_nodes):
        with open('config/participant_' + str(i) + '.yaml') as f:
            participant_config = yaml.safe_load(f)
        participant_config['network_args']['neighbors'] = topologymanager.get_neighbors_string(i)
        participant_config['scenario_args']['name'] = experiment_name
        participant_config['device_args']['idx'] = i
        participant_config["network_args"]["ip"] = topologymanager.get_node(i)[0]
        participant_config["network_args"]["port"] = topologymanager.get_node(i)[1]
        participant_config["network_args"]["ipdemo"] = topologymanager.get_node(i)[3]
        participant_config['device_args']['uid'] = hashlib.sha1((str(participant_config["network_args"]["ip"]) + str(participant_config["network_args"]["port"])).encode()).hexdigest()
        if participant_config["device_args"]["start"]:
            if not is_start_node:
                is_start_node = True
            else:
                raise ValueError("Only one node can be start node")
        with open('config/participant_' + str(i) + '.yaml', 'w') as f:
            yaml.dump(participant_config, f, sort_keys=False, default_flow_style=False, allow_unicode=True, indent=2)
        config.add_participant_config('config/participant_' + str(i) + '.yaml')
    if not is_start_node:
        raise ValueError("No start node found")

    if not args.simulation:
        for i in config.topology['nodes']:
            logging.info("[Mender.module] Device {} | IP: {} | MAC: {}".format(i['id'], i['ipdemo'], i['mac']))
            logging.info("[Mender.module] \tCreating artifacts...")
            logging.info("[Mender.module] \tSending Fedstellar framework...")
            # mender.deploy_artifact_device("my-update-2.0.mender", i['id'])
            logging.info("[Mender.module] \tSending configuration...")
            time.sleep(5)

    # Add role to the topology (visualization purposes)
    topologymanager.update_nodes(config.participants)
    topologymanager.draw_graph(save=True)

    webserver = True  # TODO: change it
    if webserver:
        from fedstellar.webserver.app import run_webserver
        logging.info("Starting webserver")

        # multiprocessing.Process(target=run_webserver, args=(config, topologymanager)).start()
        server_process = multiprocessing.Process(target=run_webserver, args=(config, topologymanager))
        server_process.start()
        time.sleep(2)
        # Export the topology configuration and the participants configuration
        topologymanager_serialized = pickle.dumps(topologymanager)
        config_serialized = pickle.dumps(config)
        import requests
        url = f'http://{config.participants[0]["scenario_args"]["controller"]}/api/topology'
        requests.post(f'http://{config.participants[0]["scenario_args"]["controller"]}/api/topology', data=topologymanager_serialized)
        requests.post(f'http://{config.participants[0]["scenario_args"]["controller"]}/api/config', data=config_serialized)

    while True:
        time.sleep(1)

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
