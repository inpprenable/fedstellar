import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime

# Add contents root directory to path
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
argparser.add_argument('-c', '--config', dest='participant_config_file', default="/Users/enrique/Documents/PhD/fedstellar/fedstellar/config/participant_config.yaml",
                    help='Path to the configuration file')
argparser.add_argument('-t', '--topology', dest='topology_config_file', default="/Users/enrique/Documents/PhD/fedstellar/fedstellar/config/topology_config_mender_dfl.json",
                    help='Path to the topology file')
argparser.add_argument('-m', '--no-mender', action='store_false', dest='mender', help='Mender for deployment')
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

    # Load the environment variables
    envpath = os.path.join(os.path.dirname(__file__), '../fedstellar/.env')
    envpath = os.path.abspath(envpath)
    load_dotenv(envpath)

    # Get some info about the backend
    # collect_env()

    # Import configuration file
    config = Config(topology_config_file=args.topology_config_file, participant_config_file=args.participant_config_file)
    logging.info("Loading participant configuration files")
    time.sleep(2)
    logging.info("Participant configuration file\n{}".format(config.get_participant_config()))

    n_nodes = len(config.topology_config['nodes'])

    fed_architecture = 'DFL'
    for n in config.topology_config['nodes']:
        if n['role'] == "server":
            fed_architecture = 'CFL'
            break

    experiment_name = f"fedstellar_{fed_architecture}_{args.name}"

    logging.info("Federated architecture: {}".format(fed_architecture))

    # Create network topology using topology manager (random)
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True,
    #                           undirected_neighbor_num=3)
    # topologymanager.generate_topology()
    # topology = topologymanager.get_topology()

    # Create a partially connected network (ring-structured network)
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True)
    # topologymanager.generate_ring_topology(increase_convergence=True)

    if fed_architecture == 'DFL':
        # Create a fully connected network
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True, undirected_neighbor_num=n_nodes - 1)
        topologymanager.generate_topology()
        # topologymanager.generate_ring_topology(increase_convergence=True)
    elif fed_architecture == 'CFL':
        # Create a centralized network
        topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True, server=True)
        topologymanager.generate_topology()
    else:
        raise ValueError("Not supported federated architecture yet")
    # topology = topologymanager.get_topology()
    # logging.info(topology)

    # Also, it is possible to use a custom topology using adjacency matrix
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, topology=[[0, 1, 1, 1], [1, 0, 1, 1]. [1, 1, 0, 1], [1, 1, 1, 0]])

    # Assign nodes to topology
    nodes_ip_port = []
    for i in config.topology_config['nodes']:
        nodes_ip_port.append((i['ip'], i['port'], i['role'], i['ipdemo']))

    topologymanager.add_nodes(nodes_ip_port)
    topologymanager.draw_graph()

    if args.mender:
        logging.info("[Mender.module] Mender module initialized")
        time.sleep(2)
        mender = Mender()
        logging.info("[Mender.module] Getting token from Mender server: {}".format(os.getenv("MENDER_SERVER")))
        mender.renew_token()
        time.sleep(2)
        logging.info("[Mender.module] Getting devices from {} with group Cluster_Thun".format(os.getenv("MENDER_SERVER")))
        time.sleep(2)
        mender.get_devices_by_group("Cluster_Thun")
        logging.info("[Mender.module] Getting a pool of devices: 5 devices")
        logging.info("Generating topology configuration file\n{}".format(config.get_topology_config()))
        time.sleep(5)
        for i in config.topology_config['nodes']:
            logging.info("[Mender.module] Device {} | IP: {} | MAC: {}".format(i['id'], i['ipdemo'], i['mac']))
            logging.info("[Mender.module] \tCreating artifacts...")
            logging.info("[Mender.module] \tSending Fedstellar framework...")
            logging.info("[Mender.module] \tSending configuration...")
            time.sleep(5)
    else:
        logging.info("Generating topology configuration file\n{}".format(config.get_topology_config()))

    # Create nodes
    python_path = '/Users/enrique/miniforge3/envs/phd-workspace/bin/python'
    start_node = False

    for idx in range(1, n_nodes):
        # logging.info("Neighbors of node " + str(idx) + ": " + str(topologymanager.get_neighbors_string(idx)))
        command = 'cd /Users/enrique/Documents/PhD/fedstellar/examples' + '; ' + python_path + ' -u node_start.py ' + str(idx) + ' ' + str(experiment_name) + ' ' + str(topologymanager.get_node(idx)[0]) + ' ' + str(topologymanager.get_node(idx)[1]) + ' ' + str(config.topology_config['nodes'][idx]['ipdemo']) + ' ' + str(n_nodes) + ' ' + str(start_node) + ' ' + str(
            config.topology_config['nodes'][idx]['role']) + ' ' + str(topologymanager.get_neighbors_string(idx)) + ' 2>&1'
        if sys.platform == "darwin":
            os.system("""osascript -e 'tell application "Terminal" to activate' -e 'tell application "Terminal" to do script "{}"'""".format(command))
        else:
            os.system(
                'cd /Users/enrique/Documents/PhD/fedstellar/examples' + ';nohup  ' + python_path + ' -u node_start.py '
                + str(idx)
                + ' ' + str(experiment_name)
                + ' ' + str(topologymanager.get_node(idx)[0])
                + ' ' + str(topologymanager.get_node(idx)[1])
                + ' ' + str(config.topology_config['nodes'][idx]['ipdemo'])
                + ' ' + str(n_nodes)
                + ' ' + str(start_node)
                + ' ' + str(config.topology_config['nodes'][idx]['role'])
                + ' ' + str(topologymanager.get_neighbors_string(idx))
                + ' 2>&1 &')

    start_node = True
    # logging.info("Neighbors of node " + str(0) + ": " + str(topologymanager.get_neighbors_string(0)))
    if sys.platform == "darwin":
        command = 'cd /Users/enrique/Documents/PhD/fedstellar/examples' + '; ' + python_path + ' -u node_start.py ' + str(0) + ' ' + str(experiment_name) + ' ' + str(topologymanager.get_node(0)[0]) + ' ' + str(topologymanager.get_node(0)[1]) + ' ' + str(config.topology_config['nodes'][0]['ipdemo']) + ' ' + str(n_nodes) + ' ' + str(start_node) + ' ' + str(
            config.topology_config['nodes'][0]['role']) + ' ' + str(topologymanager.get_neighbors_string(0)) + ' 2>&1'
        os.system("""osascript -e 'tell application "Terminal" to activate' -e 'tell application "Terminal" to do script "{}"'""".format(command))
    else:
        os.system(
            'cd /Users/enrique/Documents/PhD/fedstellar/examples' + ';nohup  ' + python_path + ' -u node_start.py '
            + str(n_nodes - 1)
            + ' ' + str(experiment_name)
            + ' ' + str(topologymanager.get_node(0)[0])
            + ' ' + str(topologymanager.get_node(0)[1])
            + ' ' + str(config.topology_config['nodes'][0]['ipdemo'])
            + ' ' + str(n_nodes)
            + ' ' + str(start_node)
            + ' ' + str(config.topology_config['nodes'][0]['role'])
            + ' ' + str(topologymanager.get_neighbors_string(0))
            + ' 2>&1 &')

    logging.info('Press Ctrl+C for exit')
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
