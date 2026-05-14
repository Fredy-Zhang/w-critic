from contextlib import contextmanager

import torch
import torch.autograd as autograd


@contextmanager
def _sdpa_math_context():
    """Force the math (non-fused) SDPA backend so create_graph=True works
    through attention layers during WGAN-GP double-backward."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            yield
    except ImportError:
        yield  # PyTorch < 2.1 — no fused backends, so math is always used


def calculate_gradient_penalty(
    discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_term: float = 10.0,
):
    """WGAN-GP gradient penalty.

    Interpolates uniformly between real and fake, evaluates the critic, then
    penalises the squared deviation of the gradient norm from 1.

    The sdpa_math_context wrapper is mandatory for Transformer/Hybrid critics:
    flash-attention and memory-efficient attention do not support the
    create_graph=True double-backward required here.

    Returns:
        penalty  : scalar tensor, ready to add to the critic loss.
        grad_norm: float, mean gradient norm over the batch (for logging /
                   early-stop monitoring).
    """
    B = min(real.size(0), fake.size(0))
    real, fake = real[:B], fake[:B]

    # Random convex combination along the batch-broadcast dimensions
    eta = torch.rand([B] + [1] * (real.ndim - 1), device=real.device).expand_as(real)
    interp = (eta * real + (1.0 - eta) * fake).requires_grad_(True)

    with _sdpa_math_context():
        prob = discriminator(interp)

    grads = autograd.grad(
        prob, interp,
        grad_outputs=torch.ones_like(prob),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grads.view(B, -1).norm(2, dim=1)          # (B,)
    penalty = ((grad_norm - 1.0) ** 2).mean() * lambda_term
    return penalty, grad_norm.mean().item()
