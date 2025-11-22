---
title: Forward Propagation in a Deep Network
parent: Week 4 - Deep Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 2
last_modified_date: 2025-11-23 10:27:00 +1100
---

# Forward Propagation in a Deep Network
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

In the [previous lesson]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-4/deep-l-layer-neural-network.md %}), we defined deep L-layer neural networks and established notation for describing them. Now we'll see how to implement **forward propagation** in deep networks.

We'll cover forward propagation in two steps:

1. **Single training example**: Compute activations for one example $x$
2. **Vectorized version**: Process entire training set simultaneously

> **Key Insight**: Forward propagation in deep networks is just repeating the same computation layer by layer!

## Forward Propagation for a Single Example

![Deep neural network architecture with 4 layers showing 3 input features x₁, x₂, x₃ on the left (layer 0), connecting through fully connected layers to a single output ŷ on the right (layer 4). The network has 5 units in layer 1, 5 units in layer 2, and 3 units in layer 3. Blue annotations indicate layer superscripts [0], [1], [2], [3], [4] and unit counts n⁰, n¹, n², n³, n⁴. Each layer is fully connected to the next layer with weighted connections shown as lines between nodes. The diagram illustrates the standard notation for deep learning where the input layer is labeled as layer 0 and subsequent layers are numbered sequentially.](/assets/images/deep-learning/neural-networks/week-4/deep_l_layer.png)

### Layer-by-Layer Computation

Given a single training example $x$, we compute activations layer by layer:

**Layer 1** (First hidden layer):

$$z^{[1]} = W^{[1]} x + b^{[1]}$$

$$a^{[1]} = g^{[1]}(z^{[1]})$$

where:

- $W^{[1]}, b^{[1]}$ are the parameters for layer 1
- $g^{[1]}$ is the [activation function]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/activation-functions.md %}) for layer 1

**Layer 2** (Second hidden layer):

$$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$$

$$a^{[2]} = g^{[2]}(z^{[2]})$$

Notice: Layer 2 uses the **activations from layer 1** as input!

**Layer 3** (Third hidden layer):

$$z^{[3]} = W^{[3]} a^{[2]} + b^{[3]}$$

$$a^{[3]} = g^{[3]}(z^{[3]})$$

**Layer 4** (Output layer):

$$z^{[4]} = W^{[4]} a^{[3]} + b^{[4]}$$

$$a^{[4]} = g^{[4]}(z^{[4]}) = \hat{y}$$

The final layer activation $a^{[4]}$ is our prediction $\hat{y}$!

### Key Observation: Replacing $x$ with $a^{[0]}$

Notice that $x$ is the input to the first layer. We can write:

$$a^{[0]} = x$$

Now all equations look uniform:

$$\boxed{\begin{align}
z^{[1]} &= W^{[1]} a^{[0]} + b^{[1]} \\
z^{[2]} &= W^{[2]} a^{[1]} + b^{[2]} \\
z^{[3]} &= W^{[3]} a^{[2]} + b^{[3]} \\
z^{[4]} &= W^{[4]} a^{[3]} + b^{[4]}
\end{align}}$$

### General Forward Propagation Equations

For any layer $l$ (where $l = 1, 2, \ldots, L$):

$$\boxed{\begin{align}
z^{[l]} &= W^{[l]} a^{[l-1]} + b^{[l]} \\
a^{[l]} &= g^{[l]}(z^{[l]})
\end{align}}$$

**Interpretation**:
- Take activations from previous layer: $a^{[l-1]}$
- Apply linear transformation: $W^{[l]} a^{[l-1]} + b^{[l]}$
- Apply activation function: $g^{[l]}(z^{[l]})$
- Get activations for current layer: $a^{[l]}$

### Complete Algorithm: Single Example

```python
def forward_propagation_single_example(x, parameters, L):
    """
    Forward propagation for a single training example

    Args:
        x: input features (n^[0], 1) vector
        parameters: dict with W1, b1, W2, b2, ..., WL, bL
        L: number of layers

    Returns:
        y_hat: prediction (scalar or vector)
        cache: dict with all z and a values for backprop
    """
    cache = {}

    # Initialize: input is activation of layer 0
    a = x
    cache['a0'] = x

    # Loop through layers
    for l in range(1, L + 1):
        # Get parameters for this layer
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']

        # Linear step
        z = np.dot(W, a) + b

        # Activation step
        if l == L:
            # Output layer: use sigmoid for binary classification
            a = sigmoid(z)
        else:
            # Hidden layers: use ReLU
            a = relu(z)

        # Store for backpropagation
        cache[f'z{l}'] = z
        cache[f'a{l}'] = a

    y_hat = a
    return y_hat, cache
```

## Vectorized Forward Propagation

### Processing All Training Examples

Now let's extend to handle $m$ training examples simultaneously using **vectorization**.

**Notation reminder**:
- Lowercase: single example ($z^{[l]}, a^{[l]}$)
- Uppercase: all examples ($Z^{[l]}, A^{[l]}$)

### Matrix Form

For the entire training set:

**Layer 1**:

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

$$A^{[1]} = g^{[1]}(Z^{[1]})$$

where:
- $X = A^{[0]}$ has shape $(n^{[0]}, m)$ - all training examples stacked as columns
- $Z^{[1]}$ has shape $(n^{[1]}, m)$
- $A^{[1]}$ has shape $(n^{[1]}, m)$

**Layer 2**:

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

$$A^{[2]} = g^{[2]}(Z^{[2]})$$

**Layer 3**:

$$Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}$$

$$A^{[3]} = g^{[3]}(Z^{[3]})$$

**Layer 4** (Output):

$$Z^{[4]} = W^{[4]} A^{[3]} + b^{[4]}$$

$$A^{[4]} = g^{[4]}(Z^{[4]}) = \hat{Y}$$

where $\hat{Y}$ is the $(n^{[4]}, m)$ matrix of predictions for all examples.

### Understanding the Matrices

**Matrix $Z^{[l]}$** stacks the $z$ vectors for all examples:

$$Z^{[l]} = \begin{bmatrix} | & | & & | \\ z^{[l](1)} & z^{[l](2)} & \cdots & z^{[l](m)} \\ | & | & & | \end{bmatrix}$$

Each column is the $z$ vector for one training example.

**Matrix $A^{[l]}$** stacks the activation vectors:

$$A^{[l]} = \begin{bmatrix} | & | & & | \\ a^{[l](1)} & a^{[l](2)} & \cdots & a^{[l](m)} \\ | & | & & | \end{bmatrix}$$

### General Vectorized Equations

For any layer $l$ (where $l = 1, 2, \ldots, L$):

$$\boxed{\begin{align}
Z^{[l]} &= W^{[l]} A^{[l-1]} + b^{[l]} \\
A^{[l]} &= g^{[l]}(Z^{[l]})
\end{align}}$$

where $A^{[0]} = X$ (the input data matrix).

### Complete Algorithm: Vectorized

```python
def forward_propagation_vectorized(X, parameters, L):
    """
    Vectorized forward propagation for all training examples

    Args:
        X: input data (n^[0], m) - m training examples
        parameters: dict with W1, b1, W2, b2, ..., WL, bL
        L: number of layers

    Returns:
        Y_hat: predictions (n^[L], m)
        caches: list of (Z, A_prev, W, b) for each layer
    """
    caches = []

    # Initialize: input is activation of layer 0
    A = X

    # Loop through layers
    for l in range(1, L + 1):
        A_prev = A

        # Get parameters for this layer
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']

        # Linear step: Z = WA + b
        Z = np.dot(W, A_prev) + b

        # Activation step
        if l == L:
            # Output layer: sigmoid for binary classification
            A = sigmoid(Z)
        else:
            # Hidden layers: ReLU
            A = relu(Z)

        # Cache for backpropagation
        cache = (Z, A_prev, W, b)
        caches.append(cache)

    Y_hat = A
    return Y_hat, caches
```

## The For Loop is Necessary!

### Iterating Through Layers

Looking at the vectorized implementation, you might notice:

```python
for l in range(1, L + 1):
    # Compute layer l activations
```

**There's a for loop!** Don't we want to avoid for loops in neural networks?

### When For Loops Are Acceptable

**Answer**: This is one case where a for loop is **perfectly fine**!

**Why?**
- The loop iterates over **layers** (1 through $L$), not training examples
- Number of layers is typically small (2-100)
- There's no way to avoid this - layers must be computed sequentially
- Layer $l$ depends on layer $l-1$, so we can't parallelize across layers

**Vectorization achieved where it matters**:
- ✅ Across training examples (the $m$ dimension)
- ✅ Within each layer (matrix operations)
- ❌ Across layers (inherently sequential)

### What We Vectorize

**Vectorized** (good for performance):

```python
Z = np.dot(W, A_prev) + b  # Processes all m examples at once
```

**Sequential** (unavoidable):

```python
for l in range(1, L + 1):  # Must compute layer 1, then 2, then 3...
```

### Summary

Forward propagation = For loop over layers (sequential) + Matrix operations within each layer (vectorized)

## Comparison: Single vs Vectorized

| Aspect | Single Example | Vectorized (All Examples) |
|--------|---------------|---------------------------|
| **Input** | $x$ (vector) | $X$ (matrix) |
| **Notation** | Lowercase: $z^{[l]}, a^{[l]}$ | Uppercase: $Z^{[l]}, A^{[l]}$ |
| **Dimensions** | $(n^{[l]}, 1)$ | $(n^{[l]}, m)$ |
| **Computation** | One example at a time | All $m$ examples at once |
| **Speed** | Slow (m iterations needed) | Fast (vectorized) |
| **Use case** | Understanding, debugging | Training, prediction |

## Dimension Check Example

Let's verify dimensions for a 4-layer network with $m = 100$ examples:

**Architecture**:
- $n^{[0]} = 3$ (input)
- $n^{[1]} = 5$ (hidden)
- $n^{[2]} = 4$ (hidden)
- $n^{[3]} = 3$ (hidden)
- $n^{[4]} = 1$ (output)

### Layer 1

$$Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}$$

**Dimensions**:
- $W^{[1]}$: $(5, 3)$
- $A^{[0]} = X$: $(3, 100)$
- $b^{[1]}$: $(5, 1)$ (broadcasts to $(5, 100)$)
- $Z^{[1]}$: $(5, 100)$ ✓

$$A^{[1]} = g^{[1]}(Z^{[1]})$$

- $A^{[1]}$: $(5, 100)$ ✓

### Layer 2

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

**Dimensions**:
- $W^{[2]}$: $(4, 5)$
- $A^{[1]}$: $(5, 100)$
- $b^{[2]}$: $(4, 1)$ (broadcasts to $(4, 100)$)
- $Z^{[2]}$: $(4, 100)$ ✓

### Complete Dimension Summary

| Layer | $W^{[l]}$ | $b^{[l]}$ | $A^{[l-1]}$ | $Z^{[l]}$ | $A^{[l]}$ |
|-------|-----------|-----------|-------------|-----------|-----------|
| 1 | $(5, 3)$ | $(5, 1)$ | $(3, 100)$ | $(5, 100)$ | $(5, 100)$ |
| 2 | $(4, 5)$ | $(4, 1)$ | $(5, 100)$ | $(4, 100)$ | $(4, 100)$ |
| 3 | $(3, 4)$ | $(3, 1)$ | $(4, 100)$ | $(3, 100)$ | $(3, 100)$ |
| 4 | $(1, 3)$ | $(1, 1)$ | $(3, 100)$ | $(1, 100)$ | $(1, 100)$ |

**Pattern**:
- $W^{[l]}$ is $(n^{[l]}, n^{[l-1]})$
- $A^{[l]}$ is $(n^{[l]}, m)$

## Relationship to Shallow Networks

### It's Just Repetition!

If forward propagation in deep networks looks familiar, that's because it is!

**What you learned in Week 3** (2-layer network):

$$\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]} \\
A^{[1]} &= g^{[1]}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} &= g^{[2]}(Z^{[2]})
\end{align}$$

**Deep networks** (L-layer network):

Just repeat the same pattern $L$ times!

### Modularity

Think of each layer as a **reusable module**:

```
Input → [Layer Module] → [Layer Module] → [Layer Module] → Output
        (Layer 1)         (Layer 2)         (Layer 3)
```

Each module does:

1. Linear transformation: $Z = WA + b$
2. Activation: $A = g(Z)$
3. Pass output to next module

## Implementation Best Practices

### 1. Always Check Dimensions

When debugging, verify matrix dimensions at each step:

```python
print(f"W{l} shape:", W.shape)
print(f"A{l-1} shape:", A_prev.shape)
print(f"Z{l} shape:", Z.shape)
print(f"A{l} shape:", A.shape)
```

**Common bugs**:
- ❌ Wrong dimension order: $(m, n)$ instead of $(n, m)$
- ❌ Missing transpose
- ❌ Shape mismatch in matrix multiplication

### 2. Cache Values for Backpropagation

Store all intermediate values:

```python
cache = {
    'Z1': Z1, 'A1': A1,
    'Z2': Z2, 'A2': A2,
    # ... for all layers
}
```

You'll need these for [backpropagation]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/gradient-descend-for-neural-networks.md %})!

### 3. Use Consistent Notation

Match your code to the mathematical notation:

```python
# Good: Clear and matches notation
Z1 = np.dot(W1, A0) + b1
A1 = relu(Z1)

# Bad: Hard to follow
temp = weights[0] @ inputs + biases[0]
output = activation(temp)
```

## Complete Example

Let's trace through forward propagation for a 4-layer network:

```python
import numpy as np

# Network architecture
layer_dims = [3, 5, 4, 3, 1]  # n^[0] through n^[4]
m = 100  # training examples
L = 4    # layers

# Initialize random parameters
parameters = initialize_parameters(layer_dims)

# Create random input data
X = np.random.randn(3, 100)

# Forward propagation
A = X  # A^[0]

print(f"Input X (A^[0]) shape: {A.shape}")  # (3, 100)

for l in range(1, L + 1):
    W = parameters[f'W{l}']
    b = parameters[f'b{l}']

    Z = np.dot(W, A) + b

    if l < L:
        A = relu(Z)  # Hidden layers
    else:
        A = sigmoid(Z)  # Output layer

    print(f"Layer {l}:")
    print(f"  W^[{l}] shape: {W.shape}")
    print(f"  Z^[{l}] shape: {Z.shape}")
    print(f"  A^[{l}] shape: {A.shape}")

Y_hat = A
print(f"\nFinal predictions shape: {Y_hat.shape}")  # (1, 100)
```

**Output**:

```
Input X (A^[0]) shape: (3, 100)
Layer 1:
  W^[1] shape: (5, 3)
  Z^[1] shape: (5, 100)
  A^[1] shape: (5, 100)
Layer 2:
  W^[2] shape: (4, 5)
  Z^[2] shape: (4, 100)
  A^[2] shape: (4, 100)
Layer 3:
  W^[3] shape: (3, 4)
  Z^[3] shape: (3, 100)
  A^[3] shape: (3, 100)
Layer 4:
  W^[4] shape: (1, 3)
  Z^[4] shape: (1, 100)
  A^[4] shape: (1, 100)

Final predictions shape: (1, 100)
```

## Key Takeaways

1. **Forward propagation formula**: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$, then $A^{[l]} = g^{[l]}(Z^{[l]})$
2. **Single example**: Use lowercase $z^{[l]}, a^{[l]}$ (vectors)
3. **Vectorized**: Use uppercase $Z^{[l]}, A^{[l]}$ (matrices with $m$ columns)
4. **Input notation**: $x = a^{[0]}$ or $X = A^{[0]}$
5. **Output notation**: $\hat{y} = a^{[L]}$ or $\hat{Y} = A^{[L]}$
6. **For loop is OK**: Iterating through layers is unavoidable and acceptable
7. **Vectorize training examples**: Process all $m$ examples simultaneously (fast!)
8. **Sequential layers**: Layers must be computed in order (inherently sequential)
9. **Same pattern repeated**: Deep networks just repeat the 2-layer logic many times
10. **Dimension checking**: Always verify matrix shapes to catch bugs
11. **Cache intermediate values**: Store $Z^{[l]}, A^{[l]}$ for backpropagation
12. **Matrix dimensions**: $W^{[l]}$ is $(n^{[l]}, n^{[l-1]})$, $A^{[l]}$ is $(n^{[l]}, m)$
13. **Broadcasting**: $b^{[l]}$ with shape $(n^{[l]}, 1)$ broadcasts to $(n^{[l]}, m)$
14. **Modular thinking**: Each layer is a reusable computation unit
15. **Implementation matches math**: Keep code notation consistent with equations
