This file will be updated with new features, bug fixes, breaking changes, and other important information as versions are updated.

## Unreleased
- Added `rnp.at(x)`: the function form of `x.at[idx]` that also accepts plain `numpy.ndarray` (returns a modified copy of the same type) and torch tensors; JAX arrays use their native `.at`.
- README: NumPy is the default backend; documented `enable_x64()` and index updates.
- Removed the repository-root `__init__.py` (invalid syntax; it stopped pytest from collecting the tests).
