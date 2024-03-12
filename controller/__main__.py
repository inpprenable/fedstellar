import argparse
import json
import os

import questionary
import requests

from controller.api_sender import login, stop_scenario, get_status_scenario, wait_until_finish, create_scenario, logout, \
    assert_server_running
from controller.json_generator import list_federation, list_dataset, list_model, list_agg_algorithm, generate_json


def folder_path_type(path):
    """Argparse type function to check if input is a valid directory path."""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory path.")
    return path


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate or execute a list of scenarios to a FedSteallar network')
    parser.add_argument("folder", type=folder_path_type, help="The folder where the scenarios are stored")
    parser.add_argument("-g", "--generate", action="store_true", default=False, help="Generate the scenarios")
    parser.add_argument("-u", "--username", type=str, help="Username to the FedStellar platform", default="admin")
    parser.add_argument("-p", "--password", type=str, help="Password to the FedStellar platform", default="admin")

    args = parser.parse_args()
    return args


class JsonOptions:
    num_nodes = 3
    federation = "DFL"
    scenario_title = ""
    device = "cpu"
    dataset = "MNIST"
    model = "MLP"
    agg_algorithm = "FedAvg"
    rounds = 10
    epochs = 3
    IID = False


def cli_generate(folder: str):
    my_json_options = JsonOptions()
    _continue = True
    scenario_id = 0
    while _continue:
        try:
            # Asking for the number of nodes
            input_data = questionary.text("How many nodes ?", default=str(my_json_options.num_nodes)).ask()
            input_data = int(input_data)
            assert 255 > my_json_options.num_nodes > 0, "Number of nodes must be between 1 and 254"
            my_json_options.num_nodes = input_data

            # Asking for federation
            input_data = questionary.text(f"Federation? {list_federation}", default=my_json_options.federation).ask()
            assert input_data in list_federation, f"Federation must be one of {list_federation}, not {input_data}"
            my_json_options.federation = input_data

            # Asking for device
            input_data = questionary.text("Device? [\"cpu\", \"gpu\"]", default=my_json_options.device).ask()
            input_data = input_data.lower()
            assert input_data in ["cpu", "gpu"], "Device must be either 'cpu' or 'gpu'"
            my_json_options.device = input_data

            # Asking for dataset
            input_data = questionary.text(f"Dataset? {list_dataset}", default=my_json_options.dataset).ask()
            assert input_data in list_dataset, f"Dataset must be one of {list_dataset}, not {input_data}"
            my_json_options.dataset = input_data

            # Asking for model
            input_data = questionary.text(f"Model? {list_model}", default=my_json_options.model).ask()
            assert input_data in list_model, f"Model must be one of {list_model}, not {input_data}"
            my_json_options.model = input_data

            # Asking for aggregation algorithm
            input_data = questionary.text(f"Aggregation Algorithm? {list_agg_algorithm}",
                                          default=my_json_options.agg_algorithm).ask()
            assert input_data in list_agg_algorithm, f"Aggregation algorithm must be one of {list_agg_algorithm}, not {input_data}"
            my_json_options.agg_algorithm = input_data

            # Asking for rounds
            input_data = questionary.text("Number of rounds?", default=str(my_json_options.rounds)).ask()
            input_data = int(input_data)
            assert input_data > 0, "Number of rounds must be greater than 0"
            my_json_options.rounds = input_data

            # Asking for epochs
            input_data = questionary.text("Number of epochs?", default=str(my_json_options.epochs)).ask()
            input_data = int(input_data)
            assert input_data > 0, "Number of epochs must be greater than 0"
            my_json_options.epochs = input_data

            # Asking for IID
            input_data = questionary.confirm("Is IID?", default=False).ask()
            my_json_options.IID = input_data

            my_json_options.scenario_title = f"scenario_{scenario_id}"
            scenario_id += 1

            print(f"configurations: {my_json_options.__dict__}")

            json_scenario = generate_json(**my_json_options.__dict__)
            with open(f"{folder}/{my_json_options.scenario_title}.json", "w") as f:
                json.dump(json_scenario, f)

            input_data = questionary.confirm("Do you want to continue?", default=True).ask()
            _continue = input_data


        except AssertionError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")


ref_set_keys = {'scenario_title', 'scenario_description', 'simulation', 'federation', 'topology', 'nodes', 'n_nodes',
                'matrix', 'dataset', 'iid', 'model', 'agg_algorithm', 'rounds', 'logginglevel', 'accelerator',
                'network_subnet', 'network_gateway', 'epochs', 'attacks', 'poisoned_node_percent',
                'poisoned_sample_percent', 'poisoned_noise_percent', 'with_reputation', 'is_dynamic_topology',
                'is_dynamic_aggregation', 'target_aggregation', 'random_geo', 'latitude', 'longitude', 'mobility',
                'mobility_type', 'radius_federation', 'scheme_mobility', 'round_frequency',
                'mobile_participants_percent', 'additional_participants', 'schema_additional_participants'}


def load_scenarios(folder: str) -> list[dict]:
    """Load the scenarios from a folder."""
    list_scenarios = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                json_scenario = json.load(f)
                print("2 ->", json_scenario.keys())
                if ref_set_keys.issubset(json_scenario.keys()):
                    list_scenarios.append(json_scenario)
    # sort all scenario by scenario_title
    list_scenarios.sort(key=lambda x: x["scenario_title"])
    return list_scenarios


if __name__ == '__main__':
    args = create_parser()
    if args.generate:
        cli_generate(args.folder)
    else:
        if assert_server_running():
            scenarios = load_scenarios(args.folder)
            print(f"{len(scenarios)} scenarios loaded")
            username, password = args.username, args.password
            s = requests.Session()
            session_cookies = login(s, username, password)
            for scenario in scenarios:
                print(scenario)
                json_return = create_scenario(s, scenario)
                scenario_name = json_return["scenario_name"]
                print(scenario_name)
                wait_until_finish(s, scenario_name, 10)
                get_status_scenario(s, scenario_name)
                stop_scenario(s, scenario_name)
            logout(s)
        else:
            print("Server not running")
