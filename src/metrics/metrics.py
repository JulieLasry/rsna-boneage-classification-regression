import torch
import torchmetrics

class Metrics():
    """
    Metrics class implementing the main
    evaluation metric: F1-score macro. This metric
    combines the recall and precision, which 
    makes it useful for evaluating a classification
    model with immbalanced classes. The macro 
    argument is based on the class F1 average.

    Args:
        num_classes (int) : Number of classes for
            the classification task, to predict. 
            The label will either be 0, 1, 2 or 3.
    """  
    def __init__(
        self,
        num_classes: int = 4
    ) -> None:
        self.metric = torchmetrics.F1Score(
            task="multiclass", 
            num_classes=num_classes, 
            average="macro"
        )
        
                   
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor
    ) -> None:
        # Preds : logits (N, C) - target : indices (N,)
        self.metric.update(preds, target)

    def compute(
        self,
    ) -> torch.Tensor:
         return self.metric.compute()

    def reset(
        self,
    ) -> None:
        return self.metric.reset()
