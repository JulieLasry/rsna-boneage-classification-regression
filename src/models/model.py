import torch.nn as nn
from monai.networks.nets import EfficientNetBN

class ClassificationModel(nn.Module):
    """
    Model class using the EfficientNetB0 backbone 
    from MONAI. This backbone has its weights 
    frozen and we unfroze only the classification
    head ones. We only do features extraction with
    no finetuning. The head of the model is one
    linear layer. The backbone is pre-trained with
    the ImageNet set. It requires RGB images (3 channels)
    and 224x224 images size. In output, there will be
    only 4 predicted classes possibilities. 

    Args:
        num_classes(int): Number of the classes of the 
            predicted label. 
    """  
    def __init__(
        self, 
        num_classes: int 
    ):
        super().__init__()
        self.backbone = EfficientNetBN(
            model_name='efficientnet-b0', 
            pretrained=True, 
            spatial_dims=2, 
            in_channels=3, 
            num_classes=num_classes
        )

        # Freeze backbones weights 
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze head of classification weights 
        for param in self.backbone._fc.parameters():
            param.requires_grad = True

        n = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Number of trainable parameters : {n}")  # 5124 (1280*4+4)

    def forward(self, x):
        return self.backbone(x)
