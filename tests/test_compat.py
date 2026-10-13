"""Backward compatibility with 0.1.0: default write guard and top-level SciPy names."""

import subprocess
import sys

import pytest


# Names 0.1.0 exported at the top level that now live under rp.scipy
_V010_SCIPY_NAMES = [
    "OptimizeResults", "RegularGridInterpolator", "beta", "betainc", "betaln", "block_diag",
    "cho_factor", "cho_solve", "dct", "dctn", "digamma", "eigh_tridiagonal", "entr", "erf", "erfc",
    "erfinv", "exp1", "expi", "expit", "expm", "expm_frechet", "expn", "factorial", "fft",
    "fresnel", "fsolve", "funm", "gamma", "gammainc", "gammaincc", "gammaln", "hessenberg",
    "hilbert", "i0e", "i1", "i1e", "idct", "idctn", "integrate", "interpolate", "log_ndtr",
    "log_softmax", "logit", "logsumexp", "lu", "lu_factor", "lu_solve", "minimize", "multigammaln",
    "ndtr", "ndtri", "optimize", "poch", "polar", "polygamma", "rsf2csf", "schur", "softmax",
    "solve_triangular", "sparse", "spatial", "special", "sqrtm", "toeplitz", "xlog1py", "xlogy",
    "zeta",
]


def test_write_guard_off_on_fresh_import():
    out = subprocess.run([sys.executable, "-W", "ignore", "-c",
                          "import RNUMPY as rp; x = rp.zeros(3); x[0] = 1.0; print(rp.ensure_differentiable, float(x[0]))"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "1.0"]


def test_write_guard_raises_when_enabled():
    import RNUMPY as rp
    rp.use_jax = False
    rp.use_torch = False
    rp.ensure_differentiable = True
    x = rp.zeros(3)
    with pytest.raises(TypeError):
        x[0] = 1.0


@pytest.mark.parametrize("name", _V010_SCIPY_NAMES)
def test_v010_top_level_scipy_names(name):
    import RNUMPY as rp
    assert getattr(rp, name) is getattr(rp.scipy, name)
