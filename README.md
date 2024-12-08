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
To install **RNUMPY**, use pip:

```bash
pip install jnp```

## Usage Examples

### Example 1: Basic Operations
```python
import RNUMPY as rnp

# Compatible with both JAX and NumPy
x = rnp.array([1, 2, 3])
y = rnp.array([4, 5, 6])

# Perform operations
dot_product = rnp.dot(x, y)
print(dot_product)  # Output will match JAX or NumPy's behavior
```

### Example 2: Special Functions
```python
import RNUMPY as rnp

x = rnp.array([1, 2, 3])
y = rnp.array([4, 5, 6])
x = rnp.array([7, 8, 9])

result = rnp.multidot(x, y, z)
print(result)
```

In the above case, this function is being called under the hood based on whether or not JAX is being used.

```python

jl = jnp.linalg
nl = np.linalg

def multi_dot(arrays, *, precision=None):
    if not rp.use_jax: return nl.multi_dot(arrays, out=precision)
    else: return jl.multi_dot(arrays, precision=precision)

```

## Ethos
RNUMPY aims to be:

1. **A Drop-In for JAX**: All function calls and API structures mimic JAX’s documentation, ensuring that users familiar with JAX can transition effortlessly.
2. **Up-to-Date**: We strive to keep this package aligned with the latest JAX updates to maintain compatibility and feature parity.
3. **JAX-first functionality**: All functions will have JAX convention and functionality. In cases where numpy functionality differs, only the JAX functionality will be accessible.

## Contributions
Contributions are welcome! If you spot inconsistencies with JAX’s API, have feature requests, or want to help maintain parity with JAX updates, feel free to open an issue or submit a pull request.

