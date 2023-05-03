import lightning as pl
import torch
import torchmetrics


class SGDOneClassSVM(pl.LightningModule):
    def __init__(
            self,
            input_dim=17,
            nu=0.1,
            lr_rate=0.01,
            metrics=None
    ):
        super().__init__()
        self.example_input_array = torch.rand(1, input_dim)
        self.input_dim = input_dim
        self.nu = nu
        self.lr_rate = lr_rate
        self.w = torch.nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        self.rho = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        if metrics is None:
            metrics = [torchmetrics.Accuracy, torchmetrics.Precision, torchmetrics.Recall, torchmetrics.F1Score, torchmetrics.AUROC]

        self.metrics = []
        if type(metrics) is list:
            try:
                for m in metrics:
                    if m == torchmetrics.AUROC:
                        self.metrics.append(m(task='multiclass', num_classes=9))
                    else:
                        self.metrics.append(m(task='multiclass', num_classes=9))
            except TypeError:
                raise TypeError("metrics must be a list of torchmetrics.Metric")

    def forward(self, x):
        return torch.matmul(x, self.w) - self.rho

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate)

    def hinge_loss(self, y):
        return torch.mean(torch.clamp(1 - y, min=0))

    def step(self, batch, phase):
        x, y_true = batch
        y_scores = self(x)
        loss = 0.5 * torch.sum(self.w ** 2) + self.nu * self.hinge_loss(y_scores)

        y_pred = (y_scores > 0).type(torch.int64)

        # This will convert the class labels to class probability vectors before passing them to the metrics.
        # Note that this is a case-specific solution and may not be best practice in general, as performance metrics may be affected if the predictions are not true probabilities.
        # TODO: Improve this
        y_pred_prob = torch.nn.functional.one_hot(y_pred, num_classes=9).float()  # Convert predictions to class probabilities

        # Log metrics
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            metric_value = metric(y_pred_prob, y_true)
            self.log(f"{phase}/{metric_name}", metric_value, prog_bar=True, logger=True)

        # Log loss
        self.log(f"{phase}/Loss", loss, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_id):
        return self.step(batch, "Train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "Validation")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "Test")
