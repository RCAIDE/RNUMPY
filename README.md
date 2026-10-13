# RNUMPY: A JAX/Numpy Package to Write JAX-Optional Code with JAX Syntax

## What is RNUMPY?
**RNUMPY** is a Python package designed to let users write code in the JAX format while seamlessly falling back to NumPy if JAX is not installed. This enables developers to leverage the benefits of JAX speed when desired without requiring all users to overcome the challenges of JAX installation. It’s especially useful in scenarios where:

- **Portability**: You want your code to be usable by others who may not have JAX installed.
- **Ease of Use**: Users with simpler computational needs may prefer to avoid installing JAX.
- **Code Consistency**: You want a single codebase to work with either library.

### Use Cases
- Sharing JAX-like code with collaborators who face installation constraints.
- Writing reusable Python libraries that can flexibly work with JAX or NumPy.
- Prototyping JAX-based projects without committing to the full ecosystem immediately.

## Installation
To install RNUMPY, use pip. **This will not install JAX**. To install JAX as well, please see the official instructions here: [JAX Install](https://github.com/jax-ml/jax?tab=readme-ov-file#installation).

```pip install rnp```

## Usage Examples

### Example: Basic Operations
Original JAX code:
```python
import jax.numpy as jnp

x = jnp.array([1, 2, 3])
y = jnp.array([4, 5, 6])

# Perform operations
dot_product = jnp.dot(x, y)
print(dot_product)
```
RNUMPY equivalent:
```python
import RNUMPY as rnp

# Compatible with both JAX and NumPy
x = rnp.array([1, 2, 3])
y = rnp.array([4, 5, 6])

# Perform operations
dot_product = rnp.dot(x, y)
print(dot_product)  # Output will match JAX
```

### Choosing the backend
RNUMPY uses NumPy by default. Switch backends at any time:
```python
import RNUMPY as rnp

rnp.use_jax = True     # JAX (requires jax)
rnp.enable_x64()       # JAX and torch default to 32-bit floats; enable 64-bit to match NumPy

rnp.use_jax = False    # back to NumPy (the default)
```

### Index updates
In-place writes such as `x[0] = 1.0` are not differentiable. Write `x = x.at[0].set(1.0)` instead, as in JAX.
Set `rnp.ensure_differentiable = True` to make in-place writes on RNUMPY arrays raise an error that shows the replacement (off by default).
For arrays that may be plain `numpy.ndarray` (which have no `.at`), use the function form:
```python
import numpy as np
import RNUMPY as rnp

x = np.zeros((4, 3))
x = rnp.at(x)[:, 1].set(2.0)   # a modified copy; x.at[:, 1].set(2.0) for a JAX array
```

## Ethos
RNUMPY aims to be:

1. **A Drop-In for JAX**: All function calls and API structures mimic JAX’s documentation, ensuring that users familiar with JAX can transition effortlessly.
2. **Up-to-Date**: We strive to keep this package aligned with the latest JAX updates to maintain compatibility and feature parity.
3. **JAX-first functionality**: All functions will have JAX convention and functionality. In cases where numpy functionality differs, only the JAX functionality will be accessible.

## Contributions
Contributions are welcome! If you spot inconsistencies with JAX’s API, have feature requests, or want to help maintain parity with JAX updates, feel free to open an issue or submit a pull request.

## RNP Todos:

RNUMPY has all the basic JAX Numpy features, except the following, which are works in progress. There are no inherent technical challenges, this project is in a beta state. We welcome pull requests to finish these area.
- ufuncs
- mgrid/ogrids, c_, r_, s_
- bitwise operators
- isinf/isnan etc..
- Set Routines
- Most of the Scipy folder
We stubbed out these functions, so you will find a NotImplementedError if you try calling these.
