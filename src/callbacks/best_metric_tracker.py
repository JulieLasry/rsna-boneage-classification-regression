import torch
import lightning as L

class BestMetricTracker(L.Callback):
    """
    Class tracking the F1-score
    during the validation epochs.
    """  
    def __init__(
        self
    ) -> None:
        self.best_val_f1: float = float("-inf")
        self.best_epoc: int = -1
        
    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        l_module: L.LightningModule) -> None:

        metrics = trainer.callback_metrics
        val_f1 = metrics.get("val_f1")

        if val_f1 is not None and not torch.isnan(val_f1) \
              and float(val_f1) > self.best_val_f1:
            self.best_val_f1 = float(val_f1)
            self.best_epoch = trainer.current_epoch
