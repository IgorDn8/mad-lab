"""Forward/backward consistency tests for MAD affine-scan implementations.

Covers the parallel-scan primitives used by the BD-LRU / H-LRU layers:

  * the triton scan kernels (``sequential`` / ``persistent`` / ``blelloch``) and
    their hand-written adjoint backward, checked against
      - a plain sequential einsum reference (forward), and
      - autograd of the differentiable ``torch.associative_scan`` reference
        (backward);
  * the layer-level ``implementation`` options, checked so that every parallel
    scan agrees (forward *and* gradient) with the reference implementation.

These require a CUDA GPU + triton; without one the tests skip cleanly.

Run with:  uv run python -m tests.test_scans
"""

import torch

from tests import run_tests
from mad.model.layers.ops.scans.triton_scans import triton_affine_scan, _reference_scan
from mad.model.layers.bdlru_sel import BDLRU_sel
from mad.model.layers.hlru_sel import HLRU_sel


TRITON_MODES = ["sequential", "persistent", "blelloch", "chunked", "auto"]
TRITON_IMPLS = [
    "triton_sequential",
    "triton_persistent",
    "triton_parallel_blelloch",
    "triton_chunked",
    "triton_auto",
]

# blelloch and chunked both reassociate the scan into fp32 matrix products, so
# they need a looser tolerance than the essentially-exact sequential /
# persistent kernels, which accumulate the recurrence step by step. `auto`
# dispatches to one of the others, so it inherits the loosest bound.
_EXACT = 1e-4
_REASSOC_FWD, _REASSOC_BWD = 5e-3, 3e-2
FWD_ATOL = {
    "sequential": _EXACT, "persistent": _EXACT,
    "blelloch": _REASSOC_FWD, "chunked": _REASSOC_FWD, "auto": _REASSOC_FWD,
}
BWD_ATOL = {
    "sequential": _EXACT, "persistent": _EXACT,
    "blelloch": _REASSOC_BWD, "chunked": _REASSOC_BWD, "auto": _REASSOC_BWD,
}
LAYER_ATOL = {  # implementation -> (fwd, grad) abs tolerance vs the reference
    "affine_scan_torch_impl": (_EXACT, _EXACT),
    "hopscan_custom": (_EXACT, _EXACT),
    "custom_hopscan_autotune": (_EXACT, _EXACT),
    "triton_sequential": (_EXACT, _EXACT),
    "triton_persistent": (_EXACT, _EXACT),
    "triton_parallel_blelloch": (_REASSOC_FWD, _REASSOC_BWD),
    "triton_chunked": (_REASSOC_FWD, _REASSOC_BWD),
    "triton_auto": (_REASSOC_FWD, _REASSOC_BWD),
}


def _cuda_ready() -> bool:
    if not torch.cuda.is_available():
        print("... CUDA unavailable, skipping")
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        print("... triton unavailable, skipping")
        return False
    return True


def _sequential_reference(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Plain loop reference: h_t = A_t @ h_{t-1} + b_t, y_t = h_t."""
    BB, T, D, _ = A.shape
    h = torch.zeros(BB, D, device=A.device, dtype=A.dtype)
    ys = []
    for t in range(T):
        h = torch.einsum("bij,bj->bi", A[:, t], h) + b[:, t]
        ys.append(h)
    return torch.stack(ys, dim=1)


def _assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float) -> None:
    err = (got - ref).abs().max().item()
    assert err <= atol, f"{name}: max abs error {err:.3e} exceeds atol {atol:.1e}"
    print(f"    {name:34s} max_err={err:.2e}  (atol={atol:.1e})")


def test_triton_scan_forward():
    """Each triton scan matches the sequential einsum reference (forward)."""
    print("Testing triton scan forward vs sequential reference...")
    if not _cuda_ready():
        return
    torch.manual_seed(0)
    for BB, T, D in [(4, 16, 1), (3, 32, 3), (2, 64, 4), (2, 48, 5)]:
        A = torch.randn(BB, T, D, D, device="cuda") * 0.3
        b = torch.randn(BB, T, D, device="cuda")
        ref = _sequential_reference(A, b)
        print(f"  BB={BB} T={T} D={D}")
        for mode in TRITON_MODES:
            y = triton_affine_scan(A.clone(), b.clone(), mode)
            _assert_close(mode, y, ref, FWD_ATOL[mode])
    print("... passed!")


def test_triton_scan_backward():
    """Hand-written triton adjoint matches autograd of the torch reference."""
    print("Testing triton scan backward vs autograd reference...")
    if not _cuda_ready():
        return
    torch.manual_seed(0)
    for BB, T, D in [(4, 16, 1), (3, 32, 3), (2, 64, 4)]:
        A0 = torch.randn(BB, T, D, D, device="cuda") * 0.3
        b0 = torch.randn(BB, T, D, device="cuda")
        grad_y = torch.randn(BB, T, D, device="cuda")
        print(f"  BB={BB} T={T} D={D}")

        # autograd ground truth (differentiable torch associative_scan)
        Ar = A0.clone().requires_grad_(True)
        br = b0.clone().requires_grad_(True)
        yr = _reference_scan(Ar, br)
        gAr, gbr = torch.autograd.grad(yr, (Ar, br), grad_y)

        for mode in TRITON_MODES:
            A = A0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            y = triton_affine_scan(A, b, mode)
            gA, gb = torch.autograd.grad(y, (A, b), grad_y)
            _assert_close(f"{mode} grad_A", gA, gAr, BWD_ATOL[mode])
            _assert_close(f"{mode} grad_b", gb, gbr, BWD_ATOL[mode])
    print("... passed!")


def _layer_fwd_grad(layer_cls, impl, x, seed, **kw):
    """Build the layer at `impl` (fixed seed for identical init), return
    (forward output, gradient w.r.t. the input) for a fixed scalar loss."""
    torch.manual_seed(seed)
    layer = layer_cls(dim=x.shape[-1], implementation=impl, **kw).cuda()
    xin = x.clone().detach().requires_grad_(True)
    y = layer(xin)
    # deterministic, non-trivial scalar loss
    (y * _layer_fwd_grad.weight).sum().backward()
    return y.detach(), xin.grad.detach()


def _run_layer_consistency(layer_cls, ref_impl, impls, seed, **kw):
    B, T, dim = 2, 48, 128
    torch.manual_seed(seed + 1)
    x = torch.randn(B, T, dim, device="cuda")
    _layer_fwd_grad.weight = torch.randn(B, T, dim, device="cuda")

    y_ref, g_ref = _layer_fwd_grad(layer_cls, ref_impl, x, seed, **kw)
    print(f"  reference implementation: {ref_impl}")
    for impl in impls:
        y, g = _layer_fwd_grad(layer_cls, impl, x, seed, **kw)
        fwd_atol, grad_atol = LAYER_ATOL[impl]
        _assert_close(f"{impl} fwd", y, y_ref, fwd_atol)
        _assert_close(f"{impl} grad", g, g_ref, grad_atol)


def test_bdlru_implementations_match():
    """All BD-LRU parallel scans agree with `orig` in forward and gradient."""
    print("Testing BD-LRU implementation consistency (fwd + bwd)...")
    if not _cuda_ready():
        return
    impls = ["affine_scan_torch_impl", "hopscan_custom"] + TRITON_IMPLS
    for window_dim, hidden_dim in [(1, 128), (3, 43)]:
        print(f"  window_dim={window_dim} hidden_dim={hidden_dim}")
        _run_layer_consistency(
            BDLRU_sel, "orig", impls, seed=0,
            hidden_dim=hidden_dim, window_dim=window_dim,
        )
    print("... passed!")


def test_hlru_implementations_match():
    """H-LRU parallel scans agree with the sequential `orig` reference.

    Companion matrices are stored in the orig right-multiply layout; hopscan /
    triton paths transpose before the left-multiply scan so all match `orig`.
    """
    print("Testing H-LRU implementation consistency (fwd + bwd)...")
    if not _cuda_ready():
        return
    impls = ["hopscan_custom"] + TRITON_IMPLS
    print("  window_dim=16 hidden_dim=64")
    _run_layer_consistency(
        HLRU_sel, "orig", impls, seed=0,
        hidden_dim=64, window_dim=16,
    )
    print("... passed!")


if __name__ == "__main__":
    run_tests()
