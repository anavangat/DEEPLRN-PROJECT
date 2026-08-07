"""Stage 2 architectures.

ASLCNN is the paper's Figure 1: four conv blocks (32-64-128-256), each
3x3 stride 1 -> BatchNorm -> ReLU -> 2x2 MaxPool; then Global Average
Pooling -> FC 512 -> ReLU -> Dropout -> FC 36.

NOTE ON SOFTMAX: the paper's figure ends in Softmax. The module returns raw
logits because nn.CrossEntropyLoss applies log-softmax internally; adding a
Softmax layer here would apply it twice and cripple the gradients. The model
is the paper's model; the softmax lives in the loss.
"""
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class ASLCNN(nn.Module):
    def __init__(self, num_classes=36, dropout=0.5, widths=(32, 64, 128, 256)):
        super().__init__()
        c_in, blocks = 3, []
        for w in widths:
            blocks.append(ConvBlock(c_in, w))
            c_in = w
        self.features = nn.Sequential(*blocks)          # 128 -> 64 -> 32 -> 16 -> 8
        self.gap = nn.AdaptiveAvgPool2d(1)              # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Linear(widths[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)

def build_model(arch="cnn", num_classes=36, dropout=0.5):
    if arch == "cnn":
        return ASLCNN(num_classes=num_classes, dropout=dropout)
    if arch == "effnet":
        import torchvision
        m = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features                 # 1280
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m
    raise ValueError(f"unknown arch {arch!r}")