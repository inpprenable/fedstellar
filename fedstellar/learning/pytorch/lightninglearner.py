# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 

import logging
import pickle
from collections import OrderedDict
import traceback
import hashlib

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import copy

from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.learner import NodeLearner
from torch.nn import functional as F

###########################
#    LightningLearner     #
###########################


class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None, logger=None):
        self.model = model
        # self.model = torch.compile(model)  # PyTorch 2.0
        self.data = data
        self.config = config
        self.logger = logger
        self.__trainer = None
        self.epochs = 1
        logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

        # FL information
        self.round = 0
        # self.local_step = 0
        # self.global_step = 0

        self.logger.log_metrics({"Round": self.round}, step=self.logger.global_step)

    def get_round(self):
        return self.round

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data

    ####
    # Model weights
    ####
    def encode_parameters(self, params=None, contributors=None, weight=None):
        if params is None:
            params = self.model.state_dict()
        array = [val.cpu().numpy() for _, val in params.items()]
        return pickle.dumps(array)

    def decode_parameters(self, data):
        try:
            params_dict = zip(self.model.state_dict().keys(), pickle.loads(data))
            return OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        except:
            raise DecodingParamsError("Error decoding parameters")

    def check_parameters(self, params):
        # Check ordered dict keys
        if set(params.keys()) != set(self.model.state_dict().keys()):
            return False
        # Check tensor shapes
        for key, value in params.items():
            if value.shape != self.model.state_dict()[key].shape:
                return False
        return True

    def set_parameters(self, params):
        try:
            self.model.load_state_dict(params)
        except:
            raise ModelNotMatchingError("Not matching models")

    def get_parameters(self):
        return self.model.state_dict()
    
    def get_hash_model(self):
        '''
        Returns:
            str: SHA256 hash of model parameters
        '''
        return hashlib.sha256(self.encode_parameters()).hexdigest()
        

    def set_epochs(self, epochs):
        self.epochs = epochs

    def fit(self):
        try:
            if self.epochs > 0:
                self.create_trainer()
                torch.autograd.set_detect_anomaly(True)
                self.__trainer.fit(self.model, self.data)
                self.__trainer = None
        except Exception as e:
            logging.error("Something went wrong with pytorch lightning. {}".format(e))
            # Log full traceback
            logging.error(traceback.format_exc())

    def interrupt_fit(self):
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    def evaluate(self):
        try:
            if self.epochs > 0:
                self.create_trainer()
                self.__trainer.test(self.model, self.data, verbose=True)
                self.__trainer = None
                # results = self.__trainer.test(self.model, self.data, verbose=True)
                # loss = results[0]["Test/Loss"]
                # metric = results[0]["Test/Accuracy"]
                # self.__trainer = None
                # self.log_validation_metrics(loss, metric, self.round)
                # return loss, metric
            else:
                return None
        except Exception as e:
            logging.error("Something went wrong with pytorch lightning. {}".format(e))
            # Log full traceback
            logging.error(traceback.format_exc())
            return None

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        self.logger.log_metrics({"Test/Loss": loss, "Test/Accuracy": metric}, step=self.logger.global_step)
        pass

    def get_num_samples(self):
        return (
            len(self.data.train_dataloader().dataset),
            len(self.data.test_dataloader().dataset),
        )

    def finalize_round(self):
        self.logger.global_step = self.logger.global_step + self.logger.local_step
        self.logger.local_step = 0
        self.round += 1
        self.logger.log_metrics({"Round": self.round}, step=self.logger.global_step)
        pass

    def create_trainer(self):
        logging.debug("[Learner] Creating trainer with accelerator: {}".format(self.config.participant["device_args"]["accelerator"]))
        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            ),
            leave=True,
        )
        self.__trainer = Trainer(
            callbacks=[RichModelSummary(max_depth=1), progress_bar],
            max_epochs=self.epochs,
            accelerator=self.config.participant["device_args"]["accelerator"],
            devices="auto" if self.config.participant["device_args"]["accelerator"] == "cpu" else "1",  # TODO: only one GPU for now
            # strategy="ddp" if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            # strategy=self.config.participant["device_args"]["strategy"] if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            logger=self.logger,
            log_every_n_steps=20,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True
        )

    def validate_neighbour_model(self, neighbour_model_param):
        avg_loss = 0
        running_loss = 0
        bootstrap_dataloader = self.data.bootstrap_dataloader()
        num_samples = 0
        neighbour_model = copy.deepcopy(self.model)
        neighbour_model.load_state_dict(neighbour_model_param)

        # enable evaluation mode, prevent memory leaks.
        # no need to switch back to training since model is not further used.
        if torch.cuda.is_available():
            neighbour_model = neighbour_model.to('cuda')
        neighbour_model.eval()

        # bootstrap_dataloader = bootstrap_dataloader.to('cuda')

        with torch.no_grad():
            for inputs, labels in bootstrap_dataloader:
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                outputs = neighbour_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                running_loss += loss.item()
                num_samples += inputs.size(0)

        avg_loss = running_loss / len(bootstrap_dataloader)
        logging.debug("[Learner.validate_neighbour]: Computed neighbor loss over {} data samples".format(num_samples))
        return avg_loss