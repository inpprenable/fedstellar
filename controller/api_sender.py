import time
from typing import Union

import requests

# Définir l'URL de votre serveur Flask
base_url = 'http://localhost:6060'


def assert_server_running() -> bool:
    """Check if the server is running."""
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.ConnectionError:
        return False


def login(session: requests.sessions.Session, username, password):
    """Login to the server and return the session cookies."""
    login_url = base_url + '/login'
    # headers = {'Content-Type': 'multipart/form-data'}
    data = {'user': username, 'password': password}
    response = session.post(login_url, data=data)
    if response.status_code == 200:
        print("Login successful")
        return response.cookies  # Renvoie les cookies de session pour les requêtes futures
    else:
        print("Login failed")
        print(response.text)
        return None


def logout(session: requests.sessions.Session, ):
    """Logout from the server."""
    logout_url = base_url + "/logout/"
    response = session.get(logout_url)
    if response.status_code == 200:
        print("Logout successful")
        session.close()
        return response.text
    else:
        print("Logout failed")
        print(response.text)
        return None


def list_scenarii(session: requests.sessions.Session, ):
    """List all the scenarios on the server."""
    scenario_api = base_url + "/api/scenario/"
    response = session.get(scenario_api)
    if response.status_code == 200:
        response_json = response.json()
        list_name_exp = [exp["name"] for exp in response_json]
        return list_name_exp
    else:
        print("Error code " + str(response.status_code))
        print(response.text)
        return None


def create_scenario(session: requests.sessions.Session, scenario: dict) -> Union[dict, None]:
    """Create a scenario on the server."""
    scenario_api = base_url + "/api/scenario/deployment/run"
    headers = {'Content-type': 'application/json'}
    response = session.post(scenario_api, json=scenario, headers=headers)
    if response.status_code == 200:
        print("Scenario created")
        return response.json()
    else:
        print("Error code " + str(response.status_code))
        print(response.text)
        return None


def stop_scenario(session: requests.sessions.Session, scenario_name: str):
    """Stop a scenario on the server."""
    scenario_api = base_url + f"/scenario/{scenario_name}/stop"
    response = session.get(scenario_api)
    if response.status_code == 200:
        print("Scenario stopped")
        return response.text
    else:
        print("Error code " + str(response.status_code))
        print(response.text)
        return None


def get_status_scenario(session: requests.sessions.Session, scenario_name: str) -> Union[None, dict]:
    """Get the status of a scenario on the server."""
    scenario_api = base_url + f"/api/scenario/{scenario_name}/monitoring"
    response = session.get(scenario_api)
    if response.status_code == 200:
        print("Scenario Monitored")
        # json["scenario_status"] into 3 states: running, completed or finished
        return response.json()
    else:
        print("Error code " + str(response.status_code))
        print(response.text)
        return None


def get_running_scenario(session: requests.sessions.Session) -> Union[None, dict]:
    """Get the status of a scenario on the server."""
    # Don't work, need to be fixed
    scenario_api = base_url + "/api/scenario/running"
    response = session.get(scenario_api)
    if response.status_code == 200:
        # json["scenario_status"] into 3 states: running, completed or finished
        return response.json()
    else:
        print("Error code " + str(response.status_code) + "in get_running_scenario")
        print(response.text)
        return None


def is_running(session: requests.sessions.Session, scenario_name: str) -> Union[bool, None]:
    """Check if a scenario is running on the server."""
    scenario_api = base_url + f"/api/scenario/{scenario_name}/running"
    response = session.get(scenario_api)
    if response.status_code == 200:
        # json["scenario_status"] into 3 states: running, completed or finished
        response = response.json()
        print("Checked")
        return not (response is None or response["status"] != "running")

    else:
        print("Error code " + str(response.status_code))
        print(response.text)
        raise ConnectionError()


def wait_until_finish(session: requests.sessions.Session, scenario_name: str, delay_time: int = 60):
    """Wait until a scenario is finished."""

    while is_running(session, scenario_name):
        time.sleep(delay_time)
    json_status = get_status_scenario(session, scenario_name)
    if json_status["scenario_status"] == "completed":
        print("Scenario is finished, it will be stopped")
        stop_scenario(session, scenario_name)


if __name__ == '__main__':
    from controller.json_generator import generate_json

    # Exemple d'utilisation
    username = 'admin'
    password = 'admin'
    algo = "DFL"
    nb_node = 3
    s = requests.Session()
    session_cookies = login(s, username, password)
    list_name_exp = list_scenarii(s)
    print(list_name_exp)

    scenario = generate_json(nb_node, algo, rounds=1)
    json_return = create_scenario(s, scenario)
    scenario_name = json_return["scenario_name"]
    print(scenario_name)
    time.sleep(5)
    # print(get_status_scenario(s, scenario_name))
    wait_until_finish(s, scenario_name, 20)
    stop_scenario(s, scenario_name)
    print(list_name_exp)
    print(get_status_scenario(s, scenario_name))

    logout(s)
