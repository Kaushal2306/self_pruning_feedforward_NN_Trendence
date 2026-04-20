import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # --- core parameters (manually replacing nn.Linear internals) ---
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # --- per-output-neuron gate scores (the pruning knob) ---
        # Initialised slightly positive so gates start open (~sigmoid(0.1)≈0.52)
        self.gate_scores = nn.Parameter(torch.full((out_features,), 0.1))

        self._init_weights()

    def _init_weights(self):
        # Kaiming uniform — same default as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound  = 1.0 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard affine transform
        out = F.linear(x, self.weight, self.bias)

        # Soft gate: scale each output neuron by its learned gate value
        gates = torch.sigmoid(self.gate_scores)  
        out   = out * gates                        

        return out

    # ------------------------------------------------------------------
    def get_sparsity(self) -> float:
        """Fraction of neurons currently gated off (gate < 0.5), in %."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            dead  = (gates < 0.5).sum().item()
        return 100.0 * dead / self.out_features

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


#  Custom Prunable Conv Layer  (same gate idea for conv)
class PrunableConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # One gate per output channel
        self.gate_scores = nn.Parameter(torch.full((out_channels,), 0.1))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out   = F.conv2d(x, self.weight, self.bias,
                         stride=self.stride, padding=self.padding)
        gates = torch.sigmoid(self.gate_scores)          # [out_channels]
        out   = out * gates.view(1, -1, 1, 1)            # broadcast [B,C,H,W]
        return out

    def get_sparsity(self) -> float:
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            dead  = (gates < 0.5).sum().item()
        return 100.0 * dead / self.out_channels

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"k={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}")



#  Batch-norm + ReLU helper  (no prunable params here — fine)
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, stride=1, padding=1):
        super().__init__(
            PrunableConv2d(in_c, out_c, k, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


#  PrunableNet — CIFAR-10 classifier (no nn.Linear anywhere)
class PrunableNet(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- feature extractor ---
        self.features = nn.Sequential(
            # Block 1
            ConvBNReLU(3,   64,  3, padding=1),
            ConvBNReLU(64,  64,  3, padding=1),
            nn.MaxPool2d(2, 2),          # 32→16
            nn.Dropout2d(0.1),

            # Block 2
            ConvBNReLU(64,  128, 3, padding=1),
            ConvBNReLU(128, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),          # 16→8
            nn.Dropout2d(0.1),

            # Block 3
            ConvBNReLU(128, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1),
            nn.MaxPool2d(2, 2),          # 8→4
        )

        # --- classifier (all PrunableLinear, zero nn.Linear) ---
        self.classifier = nn.Sequential(
            PrunableLinear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            PrunableLinear(256, num_classes),
            # No gate on the final logit layer — we want all 10 outputs active.
            # Achieved by initialising gate_scores to a large positive value:
        )

        # Keep final layer fully open
        final_layer = self.classifier[-1]
        nn.init.constant_(final_layer.gate_scores, 10.0)  # sigmoid(10)≈1.0

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)       # flatten
        x = self.classifier(x)
        return x

    # ------------------------------------------------------------------
    def get_total_sparsity(self) -> float:
        sparsities = []
        for module in self.modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                sparsities.append(module.get_sparsity())
        return sum(sparsities) / len(sparsities) if sparsities else 0.0

    def get_layer_sparsities(self) -> dict:
        info = {}
        for name, module in self.named_modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                info[name] = f"{module.get_sparsity():.1f}%"
        return info