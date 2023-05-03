import os

import lightning as pl
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix
from torchvision.models import resnet18, resnet34, resnet50

IMAGE_SIZE = 32

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

classifiers = {
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
}


class LitCIFAR10Model(pl.LightningModule):
    """
    LightningModule of ResNet for CIFAR10.
    """

    def __init__(
            self,
            out_channels=10,
            metrics=None,
            seed=None,
            implementation="scratch",
            classifier="resnet18",
    ):
        if metrics is None:
            metrics = [MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix]

        self.metrics = []
        if type(metrics) is list:
            try:
                for m in metrics:
                    self.metrics.append(m(num_classes=out_channels))
            except TypeError:
                raise TypeError("metrics must be a list of torchmetrics.Metric")

        # Set seed for reproducibility initialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.implementation = implementation
        self.classifier = classifier

        super().__init__()
        self.example_input_array = torch.rand(1, 3, 32, 32)

        self.model = self.get_model()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

        self.epoch_num_steps = {"Train": 0, "Validation": 0, "Test": 0}
        self.epoch_loss_sum = {"Train": 0.0, "Validation": 0.0, "Test": 0.0}

        self.epoch_output = {"Train": [], "Validation": [], "Test": []}
        self.epoch_real = {"Train": [], "Validation": [], "Test": []}

    def get_model(self):
        if self.implementation == "scratch":
            model = classifiers[self.classifier]
            # ResNet models in torchvision are trained on ImageNet, which has 1000 classes, and that is why they have 1000 output neurons.
            # To adapt a pre-trained ResNet model to classify images in the CIFAR-10 dataset, you need to replace the last layer (FC layer) with a new layer that has 10 output neurons.
            model.fc = torch.nn.Linear(model.fc.in_features, 10)

        elif self.implementation == "timm":
            # model = timm.create_model(
            #    cfg.model.classifier,
            #    pretrained=cfg.model.pretrained,
            #    num_classes=cfg.train.num_classes,
            # )
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        return model

    def forward(self, images):
        """ """
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(images)}")
        y_pred = self.model(images)
        return y_pred

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01,
            weight_decay=0.01,
            momentum=0.9,
            nesterov=True,
        )
        # total_steps = cfg.train.num_epochs * self.setup_steps(self)
        # scheduler = {
        #     "scheduler": WarmupCosineLR(
        #         optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
        #     ),
        #     "interval": "step",
        #     "name": "learning_rate",
        # }
        return [optimizer]

    def log_epoch_metrics_and_loss(self, phase, print_cm=True, plot_cm=True):
        # Log loss
        epoch_loss = self.epoch_loss_sum[phase] / self.epoch_num_steps[phase]
        self.log(f"{phase}Epoch/Loss", epoch_loss, prog_bar=True, logger=True)
        self.epoch_loss_sum[phase] = 0.0

        # Log metrics
        for metric in self.metrics:
            if isinstance(metric, MulticlassConfusionMatrix):
                cm = metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase]))
                print(f"{phase}Epoch/CM\n", cm) if print_cm else None
                if plot_cm:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 7))
                    ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
                    ax.set_xlabel("Predicted labels")
                    ax.set_ylabel("True labels")
                    ax.set_title("Confusion Matrix")
                    ax.set_xticks(range(10))
                    ax.set_yticks(range(10))
                    ax.xaxis.set_ticklabels([i for i in range(10)])
                    ax.yaxis.set_ticklabels([i for i in range(10)])
                    self.logger.experiment.add_figure(f"{phase}Epoch/CM", ax.get_figure(), global_step=self.epoch_global_number[phase])
                    plt.close()
            else:
                metric_name = metric.__class__.__name__.replace("Multiclass", "")
                metric_value = metric(torch.cat(self.epoch_output[phase]), torch.cat(self.epoch_real[phase])).detach()
                self.log(f"{phase}Epoch/{metric_name}", metric_value, prog_bar=True, logger=True)

            metric.reset()

        self.epoch_output[phase].clear()
        self.epoch_real[phase].clear()

        # Reset step count
        self.epoch_num_steps[phase] = 0

        # Increment epoch number
        self.epoch_global_number[phase] += 1

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
        images, labels = batch
        y_pred = self.forward(images)
        loss = self.criterion(y_pred, labels)

        # Get metrics for each batch and log them
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        self.log_metrics(phase, y_pred, labels, print_cm=False)

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
