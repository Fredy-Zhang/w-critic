import torch.nn as nn
from .utils import plain_conv, instance_norm


class Discriminator(nn.Module):
    """Strided-convolution pyramid WGAN-GP critic.

    Halves spatial dims each block until current_size == 8, then collapses
    to a scalar via a full-spatial conv.  No sigmoid — raw unbounded output
    is required for WGAN.

    3-D: spectral norm on every conv, max_channels = ndf*4.
    2-D: no spectral norm,            max_channels = ndf*8.
    """

    def __init__(self, in_channels: int, isize: int, ndf: int, ndim: int = 3):
        super().__init__()
        max_ch = ndf * 4 if ndim == 3 else ndf * 8
        spectral = (ndim == 3)

        layers = [
            plain_conv(ndim, in_channels, ndf, 4, stride=2, padding=1, spectral=spectral),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_c = ndf
        current_size = isize // 2

        # Each block halves spatial resolution.  Stop once current_size == 8
        # so the final full-kernel conv always uses kernel=8, matching the
        # isize=64 (3 blocks) and isize=128 (4 blocks) reference examples.
        while current_size > 8:
            out_c = min(in_c * 2, max_ch)
            layers += [
                plain_conv(ndim, in_c, out_c, 4, stride=2, padding=1, spectral=spectral),
                instance_norm(ndim, out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = out_c
            current_size //= 2

        # Full-spatial collapse to (B, 1, 1, 1) / (B, 1, 1)
        layers.append(plain_conv(ndim, in_c, 1, current_size, stride=1, padding=0))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).flatten(1).squeeze(1)  # (B,)
