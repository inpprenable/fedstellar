# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


"""
Module to define constants for the DFL system.
"""
import json
import logging
import yaml
from fedstellar.encrypter import AESCipher


###################
#  Global Config  #
###################


class Config:
    """
    Class to define global config for the DFL system.
    """
    topology = {}
    participant = {}

    participants = []  # Configuration of each participant (this information is stored only in the controller)

    def __init__(self, entity, topology_config_file=None, participant_config_file=None):

        self.entity = entity

        if topology_config_file is not None:
            self.set_topology_config(topology_config_file)

        if participant_config_file is not None:
            self.set_participant_config(participant_config_file)

            """
            If ```BLOCK_SIZE`` is not divisible by the block size used for symetric encryption it will be rounded to the next closest value.
            Try to strike a balance between hyper-segmentation and excessively large block size.
            """
            self.__adjust_block_size()

    def __getstate__(self):
        # Return the attributes of the class that should be serialized
        return {'topology': self.topology, 'participant': self.participant}

    def __setstate__(self, state):
        # Set the attributes of the class from the serialized state
        self.topology = state['topology']
        self.participant = state['participant']

    def get_topology_config(self):
        return json.dumps(self.topology, indent=2)

    def get_participant_config(self):
        return yaml.dump(self.participant, indent=2)

    def _set_default_config(self):
        """
        Default values are defined here.
        """
        pass

    # Read the configuration file scenario_config.yaml, and return a dictionary with the configuration
    def set_participant_config(self, participant_config):
        with open(participant_config, 'r') as stream:
            try:
                self.participant = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def set_topology_config(self, topology_config_file):
        with open(topology_config_file) as json_file:
            self.topology = json.load(json_file)

    def add_participant_config(self, participant_config):
        with open(participant_config, 'r') as stream:
            try:
                self.participants.append(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    def __adjust_block_size(self):
        if self.entity == "participant":
            rest = self.participant['BLOCK_SIZE'] % AESCipher.get_block_size()
            if rest != 0:
                new_value = self.participant['BLOCK_SIZE'] + AESCipher.get_block_size() - rest
                logging.info(
                    "[SETTINGS] Changing buffer size to %d. %d is incompatible with the AES block size.",
                    self.participant['BLOCK_SIZE'],
                    new_value,
                )
                self.participant['BLOCK_SIZE'] = new_value
        elif self.entity == "controller":
            for participant in self.participants:
                rest = participant['BLOCK_SIZE'] % AESCipher.get_block_size()
                if rest != 0:
                    new_value = participant['BLOCK_SIZE'] + AESCipher.get_block_size() - rest
                    logging.info(
                        "[SETTINGS] Changing buffer size to %d. %d is incompatible with the AES block size.",
                        participant['BLOCK_SIZE'],
                        new_value,
                    )
                    participant['BLOCK_SIZE'] = new_value
        else:
            raise Exception("Entity not supported")
