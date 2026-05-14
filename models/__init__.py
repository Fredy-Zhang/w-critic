from .cnn import Discriminator as CnnDiscriminator

# transformer and hybrid are not yet implemented; imported lazily inside build_discriminator
__all__ = ["CnnDiscriminator", "build_discriminator"]

_ARCH_CLS = {
    "cnn": CnnDiscriminator,
}


def _auto_params(arch: str, isize: int, ndim: int, ndf: int = None) -> dict:
    """Derive architecture hyperparameters from isize.

    All capacity decisions flow from isize so the model is neither under-
    nor over-parameterised at each spatial resolution.
    """
    if arch == "cnn":
        return {"ndf": ndf if ndf is not None else max(32, min(128, isize // 2))}

    if arch == "transformer":
        patch_size = isize // 4
        n_tokens = (isize // patch_size) ** ndim          # e.g. 4^3 = 64 for isize=128
        d_model = max(128, min(512, n_tokens * 4))
        return {"patch_size": patch_size, "d_model": d_model, "n_heads": 4, "n_layers": 2}

    if arch == "hybrid":
        _ndf = ndf if ndf is not None else max(32, min(64, isize // 2))
        cnn_blocks = 3
        token_grid = max(4, min(6, isize // (2 ** cnn_blocks) // 2))
        d_model = max(128, min(256, token_grid ** ndim * 2))
        return {
            "ndf": _ndf,
            "cnn_blocks": cnn_blocks,
            "token_grid": token_grid,
            "d_model": d_model,
            "n_heads": 4,
            "n_layers": 4,
        }

    raise ValueError(f"Unknown arch: {arch!r}. Choose from {list(_ARCH_CLS)}")


def build_discriminator(arch: str, ndim: int, in_channels: int, isize: int, ndf: int = None, **kwargs):
    """Instantiate a discriminator with auto-scaled defaults.

    Any keyword argument overrides the corresponding auto-scaled value, so
    callers can tune individual hyperparameters without specifying all of them.

    Returns an nn.Module with output shape (B,) — unbounded scalar, no
    sigmoid, ready for WGAN-GP training.

    Each arch module exposes the class as ``Discriminator``, making checkpoint
    loading arch-agnostic::

        from models.cnn import Discriminator
        d = Discriminator(in_channels=1, isize=128, ndf=64, ndim=3)

    Example via factory::

        build_discriminator('cnn',         ndim=3, in_channels=1, isize=128)
        build_discriminator('transformer', ndim=3, in_channels=1, isize=128, n_layers=4)
    """
    if arch == "hybrid":
        from .hybrid import Discriminator as HybridDiscriminator
        cls = HybridDiscriminator
    elif arch == "transformer":
        from .transformer import Discriminator as TransformerDiscriminator
        cls = TransformerDiscriminator
    else:
        cls = _ARCH_CLS[arch]
    params = _auto_params(arch, isize, ndim, ndf=ndf)
    params.update(kwargs)
    return cls(in_channels=in_channels, isize=isize, ndim=ndim, **params)
