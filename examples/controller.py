import os
import signal
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from fedstellar.config.config import Config
from fedstellar.utils.topologymanager import TopologyManager

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Detect ctrl+c and run killports
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    killports()
    os.system("""osascript -e 'tell application "Terminal" to quit'""") if sys.platform == "darwin" else None
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def killports(term="python"):
    # kill all the ports related to python processes
    time.sleep(1)
    command = '''kill -9 $(lsof -i @localhost:1024-65545 | grep ''' + term + ''' | awk '{print $2}')'''
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

    experiment_name = "fedstellar{}".format(str(datetime.now().strftime('%d_%m_%Y_%H_%M')))

    # Load the environment variables
    envpath = os.path.join(os.path.dirname(__file__), '../fedstellar/.env')
    envpath = os.path.abspath(envpath)
    load_dotenv(envpath)

    # Setup controller logger
    # log_file_format = f"[%(levelname)s] - %(asctime)s - CONTROLLER - : %(message)s [in %(pathname)s:%(lineno)d]"
    # file_handler = RotatingFileHandler('{}.log'.format("logs/controller"), maxBytes=10 ** 6, backupCount=40, mode='w')
    # file_handler.setFormatter(logging.Formatter(log_file_format))
    # file_handler.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG,
    #                    handlers=[
    #                        file_handler,
    #                    ])

    # Get some info about the backend
    # collect_env()

    # Import configuration file
    config = Config(topology_config_file="/Users/enrique/Documents/PhD/fedstellar/fedstellar/config/topology_config_min_cfl.json", participant_config_file="/Users/enrique/Documents/PhD/fedstellar/fedstellar/config/participant_config.yaml")

    n_nodes = len(config.topology_config['nodes'])

    # Create network topology using topology manager (random)
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True,
    #                           undirected_neighbor_num=3)
    # topologymanager.generate_topology()
    # topology = topologymanager.get_topology()

    # Create a fully connected network
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True, undirected_neighbor_num=n_nodes - 1)
    # topologymanager.generate_topology()

    # Create a partially connected network (ring-structured network)
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True)
    # topologymanager.generate_ring_topology()

    # Create a centralized network
    topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, b_symmetric=True, server=True)
    topologymanager.generate_topology()

    topology = topologymanager.get_topology()
    print(topology)

    # Also, it is possible to use a custom topology using adjacency matrix
    # topologymanager = TopologyManager(experiment_name=experiment_name, n_nodes=n_nodes, topology=[[0, 1, 1, 1], [1, 0, 1, 1]. [1, 1, 0, 1], [1, 1, 1, 0]])

    # topologymanager.draw_graph()

    # Assign nodes to topology
    nodes_ip_port = []
    for i in config.topology_config['nodes']:
        nodes_ip_port.append((i['ip'], i['port'], i['role']))

    topologymanager.add_nodes(nodes_ip_port)
    topologymanager.draw_graph()

    # Create nodes
    python_path = '/Users/enrique/miniforge3/envs/phd-workspace/bin/python'
    start_node = False

    for idx in range(1, n_nodes):
        print("Neighbors of node " + str(idx) + ": " + str(topologymanager.get_neighbors_string(idx)))
        command = 'cd /Users/enrique/Documents/PhD/fedstellar/examples' + '; ' + python_path + ' -u node_start.py ' + str(idx) + ' ' + str(experiment_name) + ' ' + str(topologymanager.get_node(idx)[0]) + ' ' + str(topologymanager.get_node(idx)[1]) + ' ' + str(n_nodes) + ' ' + str(start_node) + ' ' + str(
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
                + ' ' + str(n_nodes)
                + ' ' + str(start_node)
                + ' ' + str(config.topology_config['nodes'][idx]['role'])
                + ' ' + str(topologymanager.get_neighbors_string(idx))
                + ' 2>&1 &')

    start_node = True
    print("Neighbors of node " + str(0) + ": " + str(topologymanager.get_neighbors_string(0)))
    if sys.platform == "darwin":
        command = 'cd /Users/enrique/Documents/PhD/fedstellar/examples' + '; ' + python_path + ' -u node_start.py ' + str(0) + ' ' + str(experiment_name) + ' ' + str(topologymanager.get_node(0)[0]) + ' ' + str(topologymanager.get_node(0)[1]) + ' ' + str(n_nodes) + ' ' + str(start_node) + ' ' + str(
            config.topology_config['nodes'][0]['role']) + ' ' + str(topologymanager.get_neighbors_string(0)) + ' 2>&1'
        os.system("""osascript -e 'tell application "Terminal" to activate' -e 'tell application "Terminal" to do script "{}"'""".format(command))
    else:
        os.system(
            'cd /Users/enrique/Documents/PhD/fedstellar/examples' + ';nohup  ' + python_path + ' -u node_start.py '
            + str(n_nodes - 1)
            + ' ' + str(experiment_name)
            + ' ' + str(topologymanager.get_node(0)[0])
            + ' ' + str(topologymanager.get_node(0)[1])
            + ' ' + str(n_nodes)
            + ' ' + str(start_node)
            + ' ' + str(config.topology_config['nodes'][0]['role'])
            + ' ' + str(topologymanager.get_neighbors_string(0))
            + ' 2>&1 &')

    print('Press Ctrl+C for exit')
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
