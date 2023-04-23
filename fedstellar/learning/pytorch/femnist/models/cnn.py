import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix

IMAGE_SIZE = 28


class CNN(pl.LightningModule):
    """
    Convolutional Neural Network (CNN) to solve FEMNIST
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=62,
            metrics=None,
            lr_rate=0.001,
            momentum=0.9,
            seed=None,
    ):
        if metrics is None:
            metrics = [MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix]

        self.metrics = []
        if type(metrics) is list:
            try:
                for m in metrics:
                    self.metrics.append(m(num_classes=10))
            except TypeError:
                raise TypeError("metrics must be a list of torchmetrics.Metric")

        # Set seed for reproducibility initialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.lr_rate = lr_rate
        self.momentum = momentum

        self.conv1 = nn.Conv2d(in_channels, 32, 7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, out_channels)

        self.loss_fn = nn.CrossEntropyLoss()

        self.epoch_num_steps = {"Train": 0, "Validation": 0, "Test": 0}
        self.epoch_loss_sum = {"Train": 0.0, "Validation": 0.0, "Test": 0.0}

        self.epoch_output = {"Train": [], "Validation": [], "Test": []}
        self.epoch_real = {"Train": [], "Validation": [], "Test": []}

    def forward(self, x):
        """ """
        x = x.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        # return self.dense2(self.act(self.dense1(x)))
        return self.out(x)

    def configure_optimizers(self):
        """ """
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate, momentum=self.momentum)

    def log_epoch_metrics_and_loss(self, phase, print_cm=True):
        # Log loss
        epoch_loss = self.epoch_loss_sum[phase] / self.epoch_num_steps[phase]
        self.log(f"{phase}Epoch/Loss", epoch_loss, prog_bar=True, logger=True)
        self.epoch_loss_sum[phase] = 0.0

        # Log metrics
        for metric in self.metrics:
            if isinstance(metric, MulticlassConfusionMatrix):
                print(f"{phase}Epoch/CM\n", metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase]))) if print_cm else None
            else:
                metric_name = metric.__class__.__name__.replace("Multiclass", "")
                metric_value = metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase])).detach()
                self.log(f"{phase}Epoch/{metric_name}", metric_value, prog_bar=True, logger=True)

            metric.reset()

        self.epoch_output[phase].clear()
        self.epoch_real[phase].clear()

        # Reset step count
        self.epoch_num_steps[phase] = 0

    def log_metrics(self, phase, y_pred, y, print_cm=False):
        self.epoch_output[phase].append(y_pred.detach())
        self.epoch_real[phase].append(y.detach())

        for metric in self.metrics:
            if isinstance(metric, MulticlassConfusionMatrix):
                print(f"{phase}/CM\n", metric(y_pred, y)) if print_cm else None
            else:
                metric_name = metric.__class__.__name__.replace("Multiclass", "")
                metric_value = metric(y_pred, y)
                self.log(f"{phase}/{metric_name}", metric_value, prog_bar=True, logger=True)

    def step(self, batch, phase):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        y_pred = torch.argmax(logits, dim=1)

        # Get metrics for each batch and log them
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        self.log_metrics(phase, y_pred, y, print_cm=False)

        # Avoid memory leak when logging loss values
        self.epoch_loss_sum[phase] += loss
        self.epoch_num_steps[phase] += 1

        return loss

    def training_step(self, batch, batch_id):
        """
        Training step for the model.
        Args:
            batch:
            batch_id:

        Returns:
        """
        return self.step(batch, "Train")

    def on_train_epoch_end(self):
        self.log_epoch_metrics_and_loss("Train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, "Validation")

    def on_validation_epoch_end(self):
        self.log_epoch_metrics_and_loss("Validation")

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, "Test")

    def on_test_epoch_end(self):
        self.log_epoch_metrics_and_loss("Test")
