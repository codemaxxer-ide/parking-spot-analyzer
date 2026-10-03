"""ImageNet MobileNetV2 with a binary head; no silent random-weight fallback."""

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ParkingMobileNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = mobilenet_v2(weights=weights)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)
        self.unfrozen_blocks = 0
        self.set_fine_tuning(0)

    def set_fine_tuning(self, last_blocks: int = 0):
        if not 0 <= last_blocks <= len(self.model.features):
            raise ValueError("Invalid number of feature blocks to unfreeze.")
        self.unfrozen_blocks = last_blocks
        for parameter in self.model.features.parameters():
            parameter.requires_grad = False
        if last_blocks:
            for parameter in self.model.features[-last_blocks:].parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BatchNorm statistics must not drift during head training.
        self.model.features.eval()
        if mode and self.unfrozen_blocks:
            self.model.features[-self.unfrozen_blocks:].train()
        return self

    def forward(self, inputs):
        return self.model(inputs)
