import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class TranslationPoseNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.reduce = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * 32 * 8 * 10 + 14, 1024),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, 6),
        )
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, images, qpos):
        batch = images.shape[0]
        images = (images - self.image_mean) / self.image_std
        features = self.reduce(self.encoder(images.reshape(batch * 3, 3, 240, 320)))
        features = features.reshape(batch, 3 * 32 * 8 * 10)
        return self.head(torch.cat([features, qpos], dim=-1))
