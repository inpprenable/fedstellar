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
    topology_config = {}
    participant_config = {}

    def __init__(self, topology_config_file=None, participant_config_file=None):

        if topology_config_file is not None:
            with open(topology_config_file) as json_file:
                self.topology_config = json.load(json_file)

        if participant_config_file is None:
            # Default configuration for MNIST dataset
            self._set_default_config()
            raise Exception("Not implemented yet")
        else:
            self._set_participant_config(participant_config_file)

        """
        If ```BLOCK_SIZE`` is not divisible by the block size used for symetric encryption it will be rounded to the next closest value.
        Try to strike a balance between hyper-segmentation and excessively large block size.
        """
        rest = self.participant_config['BLOCK_SIZE'] % AESCipher.get_block_size()
        if rest != 0:
            new_value = self.participant_config['BLOCK_SIZE'] + AESCipher.get_block_size() - rest
            logging.info(
                "[SETTINGS] Changing buffer size to %d. %d is incompatible with the AES block size.",
                self.participant_config['BLOCK_SIZE'],
                new_value,
            )
            self.participant_config['BLOCK_SIZE'] = new_value

    def get_topology_config(self):
        return json.dumps(self.topology_config, indent=2)

    def get_participant_config(self):
        return yaml.dump(self.participant_config, indent=2)

    def _set_default_config(self):
        """
        Default values are defined here.
        """
        pass

    # Read the configuration file scenario_config.yaml, and return a dictionary with the configuration
    def _set_participant_config(self, participant_config):
        with open(participant_config, 'r') as stream:
            try:
                self.participant_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
