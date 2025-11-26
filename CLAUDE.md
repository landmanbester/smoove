# CLAUDE.md - Project Context for Claude Code

## Project Overview

**smoove** is a lightweight library providing miscellaneous fast smoothing utilities for scientific computing. The project implements robust smoothing algorithms using Kalman filtering, RTS smoothing, and Gaussian Process Regression (GPR) with a focus on **performance and numerical efficiency**. The project prioritizes **simplicity and minimalism** over feature completeness.

## Core Philosophy

### 1. **Simplicity First**
- Keep implementations straightforward and readable
- Avoid over-engineering solutions
- Prefer explicit over implicit behavior
- Don't add features "just in case" - wait for actual need

### 2. **Performance-Critical Dependencies**
- Minimize external dependencies where possible
- Only add dependencies when absolutely necessary
- Current dependencies: `numpy`, `scipy`, `numba`
- `pytest` for testing (version >= 6.2.2)
- Question: "Can this be done with stdlib or existing deps?" before adding a dependency

### 3. **Modern Python Best Practices**
- Python 3.10+ minimum version (transitioning from 3.7+)
- Use type hints consistently
- Follow PEP 8 style
- Prefer functional approaches over classes when possible (aligns with maintainer's preference)
- Use `@numba.njit` decorators for performance-critical numerical code

## Project Structure

```
smoove/
├── src/
│   └── smoove/
│       ├── __init__.py      # Package version info
│       └── utils.py         # Core smoothing algorithms
├── tests/
│   ├── __init__.py
│   └── test_kanterp.py      # Tests for Kalman interpolation
├── pyproject.toml           # Modern PEP 621 compliant packaging
├── README.md
├── LICENSE
└── CLAUDE.md                # Project context for Claude Code
```

**Note**: This project uses the modern **src-layout**, which is the recommended practice for Python packages. The `src/` directory contains the actual package code, which helps avoid import issues during development and testing.

## Development Tools

- **Package Manager**: `uv` (Astral) with modern `pyproject.toml`
- **Build Backend**: `uv_build` (PEP 517/621 compliant)
- **Linting/Formatting**: `ruff`
- **Testing**: `pytest` (>= 6.2.2)
- **Performance**: `numba` for JIT compilation of numerical routines

### Running Tools

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install in development mode (creates .venv and installs all dependencies)
uv sync --extra dev

# Run tests
uv run pytest tests/

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Build distributions
uv build
```

**Note**: `uv sync` creates a virtual environment (`.venv/`) and installs the package in editable mode along with all dependencies. This is the modern replacement for `pip install -e .`

## Coding Standards

### Type Hints
- Always use type hints for function signatures
- Use `from typing import Any` for generic types
- NumPy arrays should be typed with `np.ndarray` or more specific array types
- Numba-compiled functions may have limited type hint support but should still document types in docstrings

### Functions Over Classes
- Prefer pure functions where possible
- Use functions for transformations and data processing
- Only use classes when:
  - State management is truly necessary
  - You need inheritance or polymorphism
  - The maintainer explicitly requests it

### Error Handling
- Be explicit about error cases
- Provide helpful error messages
- Let exceptions propagate unless there's a good reason to catch them
- Use appropriate exceptions for numerical errors (e.g., `ValueError` for invalid inputs)

### Documentation
- Use Google-style docstrings or NumPy-style docstrings (be consistent)
- Document Args, Returns, and Raises
- Keep docstrings concise but informative
- Add comments only when code intent isn't obvious
- For numerical algorithms, document the mathematical approach when non-obvious

## What NOT to Do

❌ **Don't add these without explicit request:**
- Complex abstractions or frameworks
- Configuration systems beyond what exists
- Extensive plugin architectures
- Over-engineered validation frameworks
- Async/await unless performance demands it
- Custom metaclasses or descriptors
- Clever magic methods
- Unnecessary abstraction layers over NumPy/SciPy/Numba

❌ **Don't optimize prematurely:**
- Keep code simple first, but performance is a core concern for this library
- Profile before optimizing (use `@numba.njit` when it makes sense for numerical code)
- Balance readability with performance (both matter for scientific computing)
- Document any performance-critical optimizations

## What TO Do

✅ **Simple, direct implementations:**
```python
# Good: Simple and clear
def abs_diff(x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Compute absolute pairwise differences."""
    # Direct implementation with clear intent
    ...

# Avoid: Over-engineered
class PairwiseDistanceCalculator:
    def __init__(self, strategy: DistanceStrategy):
        self._strategy = strategy

    def calculate(self, x: np.ndarray, xp: np.ndarray) -> np.ndarray:
        return self._strategy.apply(x, xp)
```

✅ **Explicit over implicit:**
```python
# Good: Clear what's happening
@numba.njit(fastmath=True, cache=True)
def evidence_fast(theta: np.ndarray, y: np.ndarray, x: np.ndarray, Rinv: np.ndarray) -> float:
    """Compute log evidence for Kalman filter.

    Args:
        theta: Parameter vector [m0, m1, p00, p11, sigmaf, sigman]
        y: Observations
        x: Time coordinates
        Rinv: Inverse weights
    """
    # Explicit parameter unpacking
    m0, m1, p00, p11, sigmaf, sigman = theta
    # ...

# Avoid: Magic inference
def evidence_fast(theta, *args):
    """Compute evidence (parameters auto-detected)."""
    # Unclear what theta contains or what args are
```

✅ **Functional style:**
```python
# Good: Pure function
def mattern52(xx: np.ndarray, sigmaf: float, l: float) -> np.ndarray:
    """Matérn 5/2 kernel function."""
    return sigmaf**2 * np.exp(-np.sqrt(5)*xx/l) * (1 + np.sqrt(5)*xx/l + 5*xx**2/(3*l**2))

# Avoid unless needed: Stateful class
class MatternKernel:
    def __init__(self, nu: float = 2.5):
        self._nu = nu
        self._cache = {}

    def compute(self, xx: np.ndarray, sigmaf: float, l: float) -> np.ndarray:
        # Unnecessary state management
        ...
```

## Current Feature Set

smoove currently implements:

### Core Algorithms
- **Gaussian Process Regression (GPR)**: Robust smoothing with Matérn 5/2 kernel
- **Kalman Filtering**: Forward filtering for state estimation
- **RTS Smoothing**: Rauch-Tung-Striebel backward smoothing
- **Robust Kalman Interpolation** (`kanterp`): Iterative robust smoothing with Student-t noise model

### Mathematical Components
- Matérn 5/2 kernel covariance function
- Cubic spline state transition matrices (via `Afunc`, `Qfunc`)
- Weighted maximum likelihood evidence optimization
- Student-t distributed residuals for outlier robustness
- Both full matrix and optimized 2×2 matrix implementations

## Algorithm Architecture

### Main Functions in `smoove/utils.py`

**Kernel Functions:**
- `mattern52(xx, sigmaf, l)`: Matérn 5/2 kernel covariance

**Utility Functions:**
- `abs_diff(x, xp)`: Compute pairwise absolute differences
- `diag_dot(A, B)`: Diagonal of A @ B.T (numba-optimized)

**Matrix Construction:**
- `Afunc(delta, m)`: State transition matrix for cubic spline
- `Qfunc(q, delta, m)`: Process noise covariance matrix

**Kalman Filter/Smoother:**
- `Kfilter(m0, P0, x, y, H, Rinv, sigmaf)`: Forward Kalman filter
- `Kfilter_fast(...)`: Optimized 2×2 matrix version
- `RTSsmoother(m, P, x, sigmaf)`: Rauch-Tung-Striebel smoother
- `RTSsmoother_fast(...)`: Optimized 2×2 version

**Gaussian Process Regression:**
- `gpr(y, x, w, xp, theta0, nu, niter)`: GPR with Student-t residuals
- `dZdtheta(...)`: Log marginal likelihood and derivatives
- `meanf(...)`: Posterior mean function
- `meancovf(...)`: Posterior mean and covariance

**Main API:**
- `kanterp(x, y, w, niter, nu0)`: Main robust Kalman interpolation
- `kanterp2(...)`: Alternative implementation
- `kanterp3(...)`: Optimized version with simplified parameter estimation

## Performance Patterns and Numba Usage

### When to Use Numba

Use `@numba.njit` for:
- Inner loops with numerical operations
- Matrix operations not already optimized by NumPy
- Functions called repeatedly in optimization loops
- Code where Python overhead dominates

**Example pattern:**
```python
@numba.njit(fastmath=True, cache=True)
def evidence_fast(theta: np.ndarray, y: np.ndarray, x: np.ndarray, Rinv: np.ndarray) -> float:
    """Compute log evidence for Kalman filter.

    Uses numba for performance in iterative optimization.
    """
    N = x.size
    Z = 0.0
    # ... tight numerical loop
    return Z
```

### Numba Constraints

Be aware of numba limitations:
- Limited support for complex Python objects
- No support for scipy functions inside njit
- Use `with numba.objmode():` to call non-numba code (incurs overhead)
- Type inference can be tricky - explicit types help

**Example with objmode:**
```python
@numba.njit
def some_function(...):
    # numba code
    with numba.objmode(u='float64[:,:]', s='float64[:]', v='float64[:,:]'):
        u, s, v = np.linalg.svd(matrix, hermitian=True)
    # continue with numba code
```

### Optimization Strategy

1. **Write clear code first**: Get the algorithm correct
2. **Profile**: Identify actual bottlenecks
3. **Optimize selectively**: Only optimize hot paths
4. **Provide alternatives**: Keep both `func` and `func_fast` versions when possible
   - Example: `kanterp`, `kanterp2`, `kanterp3` with increasing optimization levels

### Array Operations

Prefer NumPy vectorized operations over explicit loops (except in numba functions):
```python
# Good: Vectorized
def mattern52(xx: np.ndarray, sigmaf: float, l: float) -> np.ndarray:
    return sigmaf**2 * np.exp(-np.sqrt(5)*xx/l) * (1 + np.sqrt(5)*xx/l + 5*xx**2/(3*l**2))

# Avoid: Explicit loop (unless in numba)
def mattern52_slow(xx: np.ndarray, sigmaf: float, l: float) -> np.ndarray:
    result = np.zeros_like(xx)
    for i in range(len(xx)):
        result[i] = sigmaf**2 * np.exp(-np.sqrt(5)*xx[i]/l) * ...
    return result
```

### Memory Efficiency

For large datasets:
- Pre-allocate arrays when size is known
- Avoid unnecessary copies
- Use in-place operations where appropriate
- Be mindful of intermediate array allocations

## Testing Patterns

### Numerical Testing Best Practices

Testing numerical code requires special considerations:

```python
import numpy as np
import pytest

def test_kanterp():
    """Test robust Kalman interpolation with synthetic data."""
    # 1. Define ground truth function
    def func(x):
        return 10*np.sin(20*x**2)*np.exp(-x**2/0.25) + np.exp(x)

    # 2. Set random seed for reproducibility
    np.random.seed(420)

    # 3. Generate synthetic data
    N = 128
    x = np.sort(np.random.random(N))
    f = func(x)
    sigman = np.ones(N)
    n = sigman * np.random.randn(N)
    w = 1/sigman**2
    y = f + n

    # 4. Add outliers (test robustness)
    for i in range(int(0.1*N)):
        idx = np.random.randint(0, N)
        y[idx] += 10 * np.random.randn()
        w[idx] = 0.25

    # 5. Run algorithm
    ms, Ps = kanterp(x, y, w, niter=3, nu0=2)

    # 6. Assert properties (not exact values)
    assert ms.shape == (2, N)  # State shape
    assert Ps.shape == (2, 2, N)  # Covariance shape
    assert np.all(np.isfinite(ms))  # No NaNs/Infs
    assert np.all(Ps[:, :, :] >= 0)  # Positive semi-definite
```

### Key Testing Principles

1. **Use synthetic data with known properties**
2. **Set random seeds for reproducibility**
3. **Test shape and type, not exact numerical values**
4. **Test edge cases**: empty arrays, single points, all zeros, etc.
5. **Test robustness**: outliers, missing data (zero weights)
6. **Verify mathematical properties**: positive definiteness, monotonicity, etc.

### Tolerance in Numerical Assertions

```python
# Good: Use appropriate tolerances
np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

# Avoid: Exact equality (almost always fails for floats)
assert result == expected  # Bad for floating point!
```

## Modern Tooling Setup

This project uses modern Python packaging best practices:

### Current State
- ✅ Uses `pyproject.toml` (PEP 621 compliant)
- ✅ `uv_build` as the build backend (PEP 517)
- ✅ Python 3.10+ minimum version
- ✅ `uv` for package management
- ✅ `ruff` configured for linting and formatting
- ✅ Modern src-layout structure
- ✅ CI/CD with GitHub Actions using `uv`
- ✅ `setup.py` removed (pure pyproject.toml)

### Future Improvements
- [ ] Add type hints throughout codebase
- [ ] Add more comprehensive tests
- [ ] Consider pre-commit hooks with ruff
- [ ] Add documentation (Sphinx or MkDocs)

When adding features, ask:
1. Is this feature explicitly requested?
2. Can it be implemented in <50 lines?
3. Does it require new dependencies?
4. Is there a simpler alternative?

Only proceed if answers align with project philosophy.

## Examples of Good Changes

✅ **Adding a new kernel function:**
- Small, focused change
- Clear mathematical definition
- Minimal code addition
- Follows existing patterns (like `mattern52`)

✅ **Optimizing a hot path with numba:**
- Clear performance benefit
- Profile-driven optimization
- Keep original version for reference
- Document tradeoffs

✅ **Improving numerical stability:**
- Better user experience
- No new dependencies
- Use established numerical techniques
- Add tests for edge cases

✅ **Adding type hints to existing functions:**
- Modernizes codebase
- Improves IDE support
- No runtime impact
- Aligns with project goals

## Examples of Changes to Avoid

❌ **Adding a plugin system for custom kernels:**
- Over-engineered for current needs
- Adds complexity
- No explicit request
- Simple functions work fine

❌ **Creating an abstract base class hierarchy for filters:**
- Premature abstraction
- Functional approach works fine
- Adds mental overhead
- Not needed for this library size

❌ **Implementing a custom configuration framework:**
- Way too complex
- Use function parameters directly
- Not requested
- Against minimalist philosophy

❌ **Replacing SciPy optimizers with custom implementation:**
- Unnecessary dependency removal
- SciPy is standard in scientific Python
- Would require extensive testing
- No clear benefit

## Questions to Ask Before Implementing

1. **Can this be done with stdlib or existing deps (NumPy/SciPy)?** → Use them first
2. **Can this be a simple function?** → Make it a function, not a class
3. **Is this actually needed now?** → If no, defer (YAGNI principle)
4. **Will this add dependencies?** → Avoid unless critical (keep dep tree minimal)
5. **Would the maintainer call this simple?** → If no, simplify
6. **Is the performance impact measured?** → Profile before optimizing
7. **Does this improve numerical stability/accuracy?** → Good reason to change
8. **Will this break existing user code?** → Maintain backward compatibility when possible

## Maintainer Preferences

- **Background**: Senior scientific software developer
- **Experience**: Python, NumPy, Numba, JAX, Ray, Dask
- **Style**: Functional programming preferred
- **Philosophy**: Simple, explicit, lightweight

When in doubt, prefer the simpler solution.
