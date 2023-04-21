import lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall


###############################
#    Multilayer Perceptron    #
###############################


class MLP(pl.LightningModule):
    """
    Multilayer Perceptron (MLP) to solve WADI with PyTorch Lightning.
    """

    def __init__(
            self,
            metric=[BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix],
            out_channels=1, 
            lr_rate=0.001, 
            seed=1000
    ):  # low lr to avoid overfitting

        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.example_input_array = torch.zeros(1, 123)
        self.lr_rate = lr_rate

        self.l1 = torch.nn.Linear(123, 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l3 = torch.nn.Linear(512, 256)
        self.l4 = torch.nn.Linear(256, 128)
        self.l5 = torch.nn.Linear(128, 64)
        self.l6 = torch.nn.Linear(64, 32)
        self.l7 = torch.nn.Linear(32, 16)
        self.l8 = torch.nn.Linear(16, 8)
        self.l9 = torch.nn.Linear(8, out_channels)

        self.training_step_outputs = []
        self.training_step_real = []

        self.validation_step_outputs = []
        self.validation_step_real = []

        self.test_step_outputs = []
        self.test_step_real = []
        self.metric=[]
        if type(metric) is list:
            for m in metric:
                self.metric.append(m())
        else:
            self.metric = metric()

    def forward(self, x):
        """ """
        batch_size, features = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        #x = x.view(batch_size, -1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.relu(x)
        x = self.l4(x)
        x = torch.relu(x)
        x = self.l5(x)
        x = torch.relu(x)
        x = self.l6(x)
        x = torch.relu(x)
        x = self.l7(x)
        x = torch.relu(x)
        x = self.l8(x)
        x = torch.relu(x)
        x = self.l9(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        """ """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def log_metrics(self, phase, y_pred, y, print_cm = True):
        if type(self.metric) is list:
            for m in self.metric:
                if (isinstance(m, MulticlassConfusionMatrix)):
                    if print_cm:
                        print(phase+"/CM\n", m(y_pred, y))
                    else:
                        pass
                else:
                    self.log(phase+"/"+m.__class__.__name__.replace("Multiclass", ""), m(y_pred, y))
        else:
            self.log(phase+"/"+self.metric.__class__.__name__.replace("Multiclass", ""), self.metric(y_pred, y))

    def training_step(self, batch, batch_id):
        """ """
        x, y = batch
        out = self(x)
        self.training_step_outputs.append(out)
        self.training_step_real.append(y.unsqueeze(1).float())
        loss = F.binary_cross_entropy(out, y.unsqueeze(1).float())
        self.log("Train/Loss", loss, prog_bar=True)
        self.log_metrics("Train", out, y, print_cm=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """ """
        x, y = batch
        out = self(x)
        self.validation_step_outputs.append(out)
        self.validation_step_real.append(y.unsqueeze(1).float())
        loss = F.binary_cross_entropy(out, y.unsqueeze(1).float())
        self.log("Validation/Loss", loss, prog_bar=True)
        self.log_metrics("Validation", out, y, print_cm=False)
        return loss

    def test_step(self, batch, batch_idx):
        """ """
        x, y = batch
        out = self(x)
        self.test_step_outputs.append(out)
        self.test_step_real.append(y.unsqueeze(1).float())
        loss = F.binary_cross_entropy(out, y.unsqueeze(1).float())
        self.log("Test/Loss", loss, prog_bar=True)
        self.log_metrics("Test", out, y, print_cm=False)
        return loss


    def on_train_epoch_end(self):
        out = torch.cat(self.training_step_outputs)
        y = torch.cat(self.training_step_real)
        self.log_metrics("TrainEpoch", out, y, print_cm=True)

        self.training_step_outputs.clear()  # free memory
        self.training_step_real.clear()

    def on_validation_epoch_end(self):
        out = torch.cat(self.validation_step_outputs)
        y = torch.cat(self.validation_step_real)
        self.log_metrics("ValidationEpoch", out, y, print_cm=True)

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_real.clear()
    
    def on_test_epoch_end(self):
        out = torch.cat(self.test_step_outputs)
        y = torch.cat(self.test_step_real)
        self.log_metrics("TestEpoch", out, y, print_cm=True)

        self.test_step_outputs.clear()  # free memory
        self.test_step_real.clear()