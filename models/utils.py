import torch.nn as nn


def plain_conv(ndim, in_c, out_c, kernel, stride=1, padding=0, bias=True, spectral=False):
    cls = nn.Conv3d if ndim == 3 else nn.Conv2d
    m = cls(in_c, out_c, kernel, stride=stride, padding=padding, bias=bias)
    nn.init.normal_(m.weight, 0.0, 0.02)
    nn.init.zeros_(m.bias)
    return nn.utils.spectral_norm(m) if spectral else m


def instance_norm(ndim, channels):
    cls = nn.InstanceNorm3d if ndim == 3 else nn.InstanceNorm2d
    return cls(channels, affine=True)


def plain_linear(in_f, out_f):
    m = nn.Linear(in_f, out_f)
    nn.init.normal_(m.weight, 0.0, 0.02)
    nn.init.zeros_(m.bias)
    return m
