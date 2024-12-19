import os
import pytorch_lightning as pl
import json
import torch
from torch import optim, nn
from deepnado.data.loader import TornadoDataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torchmetrics import MetricCollection, Metric, AUROC, Accuracy, Precision, Recall, ConfusionMatrix
from torch.nn.functional import sigmoid


from deepnado.models.baseline import TornadoLikelihood


# Custom wrapper to handle logits
class FromLogitsMetric(Metric):
    def __init__(self, base_metric, from_logits=True):
        super().__init__()
        self.metric = base_metric
        self.from_logits = from_logits

    def update(self, y_pred, y_true):
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
        self.metric.update(y_pred, y_true)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


# Define F1Score
class F1Score(Metric):
    def __init__(self, from_logits=True):
        super().__init__()
        self.precision = FromLogitsMetric(Precision(task="binary"), from_logits)
        self.recall = FromLogitsMetric(Recall(task="binary"), from_logits)

    def update(self, y_pred, y_true):
        self.precision.update(y_pred, y_true)
        self.recall.update(y_pred, y_true)

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        return 2 * (precision * recall) / (precision + recall + 1e-7)

    def reset(self):
        self.precision.reset()
        self.recall.reset()


class ConfusionMatrixMetrics(Metric):
    def __init__(self, task="binary", threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.conf_matrix = ConfusionMatrix(task=task)

    def update(self, y_pred, y_true):
        # Apply threshold if using probabilities/logits
        y_pred_binary = (y_pred >= self.threshold).int()
        self.conf_matrix.update(y_pred_binary, y_true)

    def compute(self):
        conf_matrix = self.conf_matrix.compute()
        tn, fp, fn, tp = conf_matrix.ravel()  # Unpack the confusion matrix
        return {
            "TruePositives": tp.item(),
            "FalsePositives": fp.item(),
            "TrueNegatives": tn.item(),
            "FalseNegatives": fn.item()
        }

    def reset(self):
        self.conf_matrix.reset()

class LightningWrapper(pl.LightningModule):
    """
    Fits tornado classifier
    """
    def __init__(self, config):
        super().__init__()
        self.model = None
        if config["model"] == "baseline":
            self.model = TornadoLikelihood()
        self.lr = config["learning_rate"]
        self.label_smoothing = config["label_smooth"]
        self.weight_decay=config["weight_decay"]
        self.lr_decay_rate=config["lr_decay_rate"]
        self.lr_decay_steps=config["lr_decay_steps"]
        if config["loss"] == "cce":
            self.loss = nn.CrossEntropyLoss(label_smoothing=config["label_smooth"]) # this should maybe be BCELoss?
        elif config["loss"] == "hinge":
            self.loss = nn.HingeEmbeddingLoss() # probably need to convert labels to -1, 1 if using this?
        elif config["loss"] == "mae":
            self.loss = nn.L1Loss()
        metrics = MetricCollection({
                "AUC": FromLogitsMetric(AUROC(task="binary"), from_logits=True),
                "AUCPR": FromLogitsMetric(AUROC(task="binary", average="macro"), from_logits=True),
                "BinaryAccuracy": FromLogitsMetric(Accuracy(task="binary"), from_logits=True),
                "ConfusionMatrix": ConfusionMatrixMetrics(task="binary"),
                "Precision": FromLogitsMetric(Precision(task="binary"), from_logits=True),
                "Recall": FromLogitsMetric(Recall(task="binary"), from_logits=True),
                "F1": F1Score(from_logits=True)
            }) 
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self,batch):
        return self.model(batch)
        
    def training_step(self, batch, _):
        batch = batch[0]
        y = torch.squeeze(batch.pop('label')) # [batch]
        logits = self.model(batch) # [batch,1,L,W] 
        logits = F.max_pool2d(logits, kernel_size=logits.size()[2:]) # [batch,1,1,1] 
        logits = torch.cat( (-logits,logits),axis=1)  # [batch,2,1,1] 
        logits = torch.squeeze(logits) # [batch,2] for binary classification
        loss = self.loss(logits, y)
        
        # Logging..
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.train_metrics:
            met_out = self.train_metrics(logits[:,1],y)
            self.log_dict(met_out, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self,batch,_):
        y,logits,loss=self._shared_eval(batch[0])
        
        # Logging..
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.valid_metrics:
            self._log_metrics(self.valid_metrics,y,logits)
        return loss
    
    def test_step(self,batch,_):
        y,logits,loss=self._shared_eval(batch)
        
        # Logging..
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.test_metrics:
            self._log_metrics(self.test_metrics,y,logits)
        return loss

    def _shared_eval(self,batch):
        y = torch.squeeze(batch['label']) # [batch]
        logits = self.model(batch) # [batch,1,L,W] 
        logits = F.max_pool2d(logits, kernel_size=logits.size()[2:]) # [batch,1,1,1] 
        logits = torch.cat( (-logits,logits),axis=1)  # [batch,2,1,1] 
        logits = torch.squeeze(logits) # [batch,2] for binary classification
        loss = self.loss(logits, y)
        return y,logits,loss
    
    def _log_metrics(self,metrics,y,logits):
        met_out = metrics(logits[:,1],y)
        self.log_dict(met_out, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        
        # Define the learning rate scheduler
        scheduler = {
            'scheduler': StepLR(optimizer, 
                                step_size=self.lr_decay_steps, 
                                gamma=self.lr_decay_rate),
            'interval': 'epoch',  # Adjust the learning rate at the end of each epoch
            'frequency': 1  # Apply the scheduler every epoch
        }
        return [optimizer], [scheduler]



def validate_log_config(config):
    """
    Asserts that the config has reasonable things in it.
    """
    model_options = ["baseline"]
    loss_options = ["cce", "hinge", "mae"]
    head_options = ["maxpool"]
    assert config["model"] in model_options, f'Unknown model type {config["model"]}. Allowed options are {model_options}'
    assert config["batch_size"] >= 1 and config["batch_size"] <= 2048
    assert isinstance(config["train_years"], list) and all(isinstance(item, int) for item in config["train_years"])
    assert isinstance(config["val_years"], list) and all(isinstance(item, int) for item in config["val_years"])
    assert 1e-10 <= config["learning_rate"] <= 5.0
    assert config["loss"] in loss_options, f'Unknown loss type {config["loss"]}. Allowed options are {loss_options}'
    assert config["head"] in head_options, f'Unknown head type {config["head"]}. Allowed options are {head_options}'
    assert isinstance(config["job_name"], str) and not config["job_name"].strip() == "", "Please provide a valid string experiment name."
    return config

def train(logger, data_root, training_config):
    logger.debug("Training model...")
    logger.debug(f"Data root is configured to be {data_root}")
    logger.debug(f"Training config path is: {training_config}")
    # Read the config file
    with open(training_config) as f:
        config = validate_log_config(json.load(f))

    # Make the dataset and data loader
    logger.debug("Instantiating data loaders")
    weights = {"wN": config["wN"], "w0": config["w0"], 
               "w1": config["w1"], "w2": config["w2"], "wW": config["wW"]}
    train_loader = TornadoDataLoader().get_dataloader(data_root, 
                                                      data_type="test",#"train", TODO Change back - just for local debug
                                                      years=config["train_years"],
                                                      batch_size=config["batch_size"],
                                                      weights=weights,
                                                      workers=config["nworkers"])
    val_loader = TornadoDataLoader().get_dataloader(data_root, 
                                                      data_type="test", 
                                                      years=config["val_years"],
                                                      batch_size=config["batch_size"],
                                                      workers=config["nworkers"])

    # Instantiate Model
    logger.debug("Instantiating model")
    model = LightningWrapper(config)

    # Instantiate Pytorch lightning trainer 
    logger.debug("Beginning training")
    tb_logger = pl.loggers.TensorBoardLogger(config["log_dir"], name=config["job_name"])
    mlflow_logger = pl.loggers.MLFlowLogger(experiment_name=config["mlflow_name"], run_name=config["job_name"], tracking_uri="http://mnemosyne.local:5555/")
    mlflow_logger.log_hyperparams(config)
    trainer = pl.Trainer(logger=[tb_logger, mlflow_logger], accelerator='auto', num_sanity_val_steps=0)

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.debug("Completed Training.")
