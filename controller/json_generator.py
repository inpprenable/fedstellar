from typing import Union

from bs4 import BeautifulSoup

list_dataset = ["MNIST", "CIFAR10", "FashionMNIST", "SYSCALL"]
list_model = ["MLP", "CNN"]
list_agg_algorithm = ["FedAvg", "Krum", "Median", "TrimmedMean"]
list_federation = ["DFL", "SDFL", "CFL"]


def node_role(i: int, federation: str):
    if federation == "DFL":
        return "aggregator"
    elif federation == "SDFL":
        return "aggregator" if i == 0 else "trainer"
    elif federation == "CFL":
        return "server" if i == 0 else "trainer"
    else:
        raise ValueError(f"Federation {federation} not supported")


def generate_json(num_nodes: int, federation: str, device: str = "cpu", dataset: str = "MNIST", model: str = "MLP",
                  agg_algorithm: str = "FedAvg", rounds: int = 10, epochs: int = 3,
                  IID: bool = False, scenario_title: str = ""
                  ) -> dict:
    assert 255 > num_nodes > 0, "Number of nodes must be between 1 and 254"
    assert device in ["cpu", "gpu"], "Device must be either 'cpu' or 'gpu'"
    assert dataset in list_dataset, f"Dataset must be one of {list_dataset}, not {dataset}"
    assert federation in list_federation, f"Federation must be one of {list_federation}, not {federation}"
    assert model in list_model, f"Model must be one of {list_model}, not {model}"
    assert agg_algorithm in list_agg_algorithm, f"Aggregation algorithm must be one of {list_agg_algorithm}, not {agg_algorithm}"
    assert rounds > 0, "Number of rounds must be greater than 0"
    assert epochs > 0, "Number of epochs must be greater than 0"
    scenario = {
        "scenario_title": scenario_title,
        "scenario_description": "",
        "simulation": True,
        "federation": "DFL",
        "topology": "Custom",
        "nodes": {},
        "n_nodes": num_nodes,
        "matrix": [[int(i != j) for j in range(num_nodes)] for i in range(num_nodes)],
        "dataset": dataset,
        "iid": IID,
        "model": model,
        "agg_algorithm": agg_algorithm,
        "rounds": rounds,
        "logginglevel": False,
        "accelerator": device,
        "network_subnet": "192.168.50.0/24",
        "network_gateway": "192.168.50.254",
        "epochs": epochs,
        "attacks": "No Attack",
        "poisoned_node_percent": "0",
        "poisoned_sample_percent": "0",
        "poisoned_noise_percent": "0",
        "with_reputation": False,
        "is_dynamic_topology": False,
        "is_dynamic_aggregation": False,
        "target_aggregation": False,
        "random_geo": True,
        "latitude": "38.023522",
        "longitude": "-1.174389",
        "mobility": False,
        "mobility_type": "both",
        "radius_federation": "1000",
        "scheme_mobility": "random",
        "round_frequency": "1",
        "mobile_participants_percent": "50",
        "additional_participants": [],
        "schema_additional_participants": "random"
    }

    for i in range(num_nodes):
        node = {
            "id": i,
            "ip": "192.168.50." + str(i + 1),
            "port": "45000",
            "role": node_role(i, federation),
            "malicious": False,
            "start": i == 0
        }
        scenario["nodes"][str(i)] = node

    return scenario


def get_scenario_name_from_html(html: str) -> Union[str, None]:
    soup = BeautifulSoup(html, 'html.parser')
    scenario_name_element = soup.find(id='scenario_name')
    if scenario_name_element is None:
        return None
    return scenario_name_element.text


if __name__ == '__main__':
    with open("retour.html", "r") as f:
        html = f.read()
        print(get_scenario_name_from_html(html))
