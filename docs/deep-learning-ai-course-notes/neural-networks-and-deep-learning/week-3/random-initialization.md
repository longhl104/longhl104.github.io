---
title: Random Initialization
parent: Week 3 - Shallow Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 10
last_modified_date: 2025-11-23 09:45:00 +1100
---

# Random Initialization
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

When training a neural network, **weight initialization** is critical for successful learning. Unlike logistic regression (where initializing weights to zero works fine), neural networks require **random initialization** of weights. This lesson explains why zero initialization fails and how to initialize properly.

> **Key Point**: Biases can be initialized to zero, but weights must be random!

## Why Not Initialize to Zero?

### The Problem: Symmetry

Let's examine what happens with zero initialization using a simple example:

![Neural network diagram showing what happens when weights are initialized to zero. At top: question "What happens if you initialize weights to zero?". Left side shows a two-layer neural network with two input nodes x1 and x2, two hidden layer nodes a1[1] and a2[1], and one output node a1[2] producing prediction y-hat. All weights W[1] and W[2] are shown as zero matrices in brackets. Annotations indicate h[1]=2 and h[2]=1. Below the diagram are mathematical notations: W[1] is a 2x2 zero matrix, b[1] is a 2x1 zero vector, showing that a1[1] equals a2[1], notation dW with matrix of u and v values, and the gradient descent update rule W[1] = W[1] - alpha*dW. Right side shows a "Symmetric" diagram where inputs x1, x2, x3 connect to a column of zeros representing hidden units with all connections having zero weights, demonstrating that all hidden units remain identical and W[1] remains a matrix of all zeros even after updates.](/assets/images/deep-learning/neural-networks/week-3/zero_initialization_problem.png)

**Network architecture:**

- Input features: $n^{[0]} = 2$
- Hidden units: $n^{[1]} = 2$
- Output units: $n^{[2]} = 1$

**Zero initialization:**

$$W^{[1]} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad b^{[1]} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$W^{[2]} = \begin{bmatrix} 0 & 0 \end{bmatrix}, \quad b^{[2]} = 0$$

### What Goes Wrong?

**Forward propagation:**

$$Z^{[1]} = W^{[1]} X + b^{[1]} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$A^{[1]} = g(Z^{[1]}) = \begin{bmatrix} g(0) \\ g(0) \end{bmatrix}$$

**Result**: Both hidden units compute **identical** activations!

$$a_1^{[1]} = a_2^{[1]}$$

This means:

- Hidden unit 1 and hidden unit 2 compute the **same function**
- They have the **same influence** on the output
- They are completely **symmetric**

### The Symmetry Persists During Training

**Backpropagation:**

Because the hidden units are symmetric, their gradients are also identical:

$$dZ_1^{[1]} = dZ_2^{[1]}$$

This means the gradient matrix $dW^{[1]}$ has identical rows:

$$dW^{[1]} = \begin{bmatrix} \text{same values} \\ \text{same values} \end{bmatrix}$$

**Weight update:**

$$W^{[1]} := W^{[1]} - \alpha \, dW^{[1]}$$

After the update, $W^{[1]}$ **still has identical rows**!

### Proof by Induction

We can prove that symmetry persists forever:

**Base case (iteration 0)**:

- Both hidden units are identical: $w_1 = w_2 = 0$

**Inductive step**:

- If hidden units are identical at iteration $t$, then:
  - They compute the same function
  - They produce the same gradients
  - Weight updates keep them identical
  - They remain identical at iteration $t+1$

**Conclusion**: No matter how long you train, hidden units remain symmetric!

### Why This Is Bad

If all hidden units compute the same function, then having $n^{[1]}$ hidden units is **no better than having just 1 hidden unit**!

$$\text{Multiple identical units} = \text{Wasted computation}$$

The network cannot learn diverse features, which is the whole point of having multiple hidden units.

## The Solution: Random Initialization

![Neural network diagram showing random weight initialization to break symmetry. At top: two input features x1 and x2 connect to two hidden layer nodes a1[1] and a2[1], which then connect to output node a1[2] producing y-hat. Blue handwritten annotations show: arrow from x1 to a1[1] labeled with W[1], arrow from x2 to a2[1] also labeled with W[1], and arrow from hidden layer to output labeled with W[2]. Below the diagram are mathematical expressions: W[1] equals np.random.randn((2,2)) times 0.01, question mark asking 100?, b[1] equals np.zeros((2,1)), W[2] equals np.random.randn((1,2)) times 0.01, and b[2] equals 0. To the right: arrow pointing to z[1] equals W[1] times X plus b[1], with a[2] equals g[2](z[2]) written below and annotated with blue arrow. Bottom right shows a sigmoid activation function curve with arrows indicating input approaching from negative infinity rises gradually, crosses origin, then approaches positive infinity, demonstrating the nonlinear activation transformation.](/assets/images/deep-learning/neural-networks/week-3/random_initialization_solution.png)

### Breaking Symmetry

To make different hidden units learn different functions, initialize weights **randomly**:

```python
# Initialize weights randomly (small values)
W1 = np.random.randn(n1, n0) * 0.01
b1 = np.zeros((n1, 1))  # Biases can be zero

W2 = np.random.randn(n2, n1) * 0.01
b2 = np.zeros((n2, 1))  # Biases can be zero
```

### Why This Works

**Random weights** → Different initial values → Different computations → **Symmetry broken**!

$$W^{[1]} = \begin{bmatrix} 0.0053 & -0.0023 \\ 0.0097 & 0.0041 \end{bmatrix}$$

Now:

- $w_1 \neq w_2$ → Different weights for each unit
- $a_1^{[1]} \neq a_2^{[1]}$ → Different activations
- $dW_1 \neq dW_2$ → Different gradients
- Units evolve differently during training ✅

### Why Biases Can Be Zero

**Important distinction:**

$$b^{[1]} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \quad \text{← This is okay!}$$

**Why?** As long as $W^{[1]}$ is random, the hidden units compute different functions:

$$z_1^{[1]} = w_1^T x + 0 \neq w_2^T x + 0 = z_2^{[1]}$$

The symmetry is already broken by different $w_1$ and $w_2$, so bias initialization doesn't matter.

## Why Small Random Values?

### The Scaling Factor: 0.01

You might wonder: why multiply by `0.01`? Why not `100` or `1000`?

```python
W1 = np.random.randn(n1, n0) * 0.01  # Why 0.01?
```

### Reason 1: Avoiding Saturation

For **sigmoid** and **tanh** activation functions:

**Forward propagation:**

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

$$A^{[1]} = g(Z^{[1]})$$

**If $W$ is too large** → $Z$ has very large/small values → Activations saturate!

#### Sigmoid Saturation

![Sigmoid saturation diagram](sigmoid-saturation.png)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Problem regions:**

- When $z \gg 0$: $\sigma(z) \approx 1$, $\sigma'(z) \approx 0$
- When $z \ll 0$: $\sigma(z) \approx 0$, $\sigma'(z) \approx 0$

**Gradient in saturated region:**

$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \approx 0$$

**Result**: Vanishing gradients → **Very slow learning**!

#### Tanh Saturation

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Problem regions:**

- When $z \gg 0$: $\tanh(z) \approx 1$, $\tanh'(z) \approx 0$
- When $z \ll 0$: $\tanh(z) \approx -1$, $\tanh'(z) \approx 0$

**Gradient in saturated region:**

$$\tanh'(z) = 1 - \tanh^2(z) \approx 0$$

### Summary: Large Weights → Slow Learning

**Chain of events:**

$$\text{Large } W \rightarrow \text{Large } |Z| \rightarrow \text{Saturated activations} \rightarrow \text{Small gradients} \rightarrow \text{Slow learning}$$

**Solution:** Initialize with small values (like `0.01`) to keep $Z$ in the responsive region.

### Reason 2: Output Layer Concerns

For **binary classification**, the output uses sigmoid:

$$\hat{y} = \sigma(z^{[L]}) = \sigma(W^{[L]} a^{[L-1]} + b^{[L]})$$

If $W^{[L]}$ is large → $z^{[L]}$ is large → Output saturates at 0 or 1 immediately → No learning!

### When Is This Less Critical?

For **ReLU activation functions**, saturation is less of an issue:

$$\text{ReLU}(z) = \max(0, z)$$

**Why?**

- No saturation for $z > 0$ (gradient is always 1)
- Only "dies" for $z < 0$ (gradient is 0)

But small initialization is still generally recommended!

## Complete Initialization Example

```python
import numpy as np

def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize parameters with random weights and zero biases
    
    Args:
        n_x: size of input layer
        n_h: size of hidden layer
        n_y: size of output layer
    
    Returns:
        parameters: dictionary containing W1, b1, W2, b2
    """
    # Random initialization for weights (small values)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return parameters

# Example usage
n_x = 3  # input features
n_h = 4  # hidden units
n_y = 1  # output units

params = initialize_parameters(n_x, n_h, n_y)

print("W1 shape:", params["W1"].shape)  # (4, 3)
print("b1 shape:", params["b1"].shape)  # (4, 1)
print("W2 shape:", params["W2"].shape)  # (1, 4)
print("b2 shape:", params["b2"].shape)  # (1, 1)

print("\nW1 sample values:")
print(params["W1"])  # Small random values around 0
```

## Comparison: Zero vs Random Initialization

| Aspect | Zero Initialization | Random Initialization |
|--------|--------------------|-----------------------|
| **Symmetry** | All hidden units identical | All hidden units different |
| **Learning** | No learning (units stay same) | Successful learning |
| **Gradients** | All rows identical | Different for each unit |
| **Effective units** | Only 1 (others redundant) | All $n^{[1]}$ units useful |
| **Feature diversity** | ❌ No diverse features | ✅ Learns diverse features |
| **Use case** | ❌ Never use for NN | ✅ Always use for NN |

## Advanced: Better Constants Than 0.01

### For Shallow Networks

For networks with **one hidden layer** (shallow networks), `0.01` works well:

```python
W = np.random.randn(n_out, n_in) * 0.01  # Good for shallow networks
```

### For Deep Networks

For **very deep networks** (many layers), you might need different initialization strategies:

**Xavier Initialization** (for sigmoid/tanh):

$$W^{[l]} = \text{np.random.randn}(n^{[l]}, n^{[l-1]}) \times \sqrt{\frac{1}{n^{[l-1]}}}$$

**He Initialization** (for ReLU):

$$W^{[l]} = \text{np.random.randn}(n^{[l]}, n^{[l-1]}) \times \sqrt{\frac{2}{n^{[l-1]}}}$$

These scale the initialization based on layer size to maintain stable gradients.

> **Note**: We'll cover advanced initialization strategies in next week's material on deep neural networks!

## Visualization: Symmetry Breaking

### Zero Initialization (Bad)

```
Before training:
Hidden Unit 1: [0, 0] → same function
Hidden Unit 2: [0, 0] → same function

After 1000 iterations:
Hidden Unit 1: [0.5, 0.3] → STILL same function
Hidden Unit 2: [0.5, 0.3] → STILL same function

Result: Wasted capacity! ❌
```

### Random Initialization (Good)

```
Before training:
Hidden Unit 1: [0.01, -0.02] → different function
Hidden Unit 2: [0.03,  0.01] → different function

After 1000 iterations:
Hidden Unit 1: [0.8, -0.4] → detects feature A
Hidden Unit 2: [0.2,  0.9] → detects feature B

Result: Diverse features learned! ✅
```

## Implementation Checklist

When initializing your neural network:

- [ ] Initialize **weights randomly** using `np.random.randn()`
- [ ] Multiply by small constant (typically `0.01`)
- [ ] Initialize **biases to zero** using `np.zeros()`
- [ ] Verify dimensions match network architecture
- [ ] For shallow networks: use `0.01`
- [ ] For deep networks: consider Xavier/He initialization (coming later)
- [ ] **Never** initialize all weights to zero
- [ ] **Never** initialize all weights to the same value

## What's Next

Congratulations! You now understand:

- How to set up a neural network with one hidden layer
- How to initialize parameters correctly
- How to compute predictions using forward propagation
- How to compute derivatives using backpropagation
- How to implement gradient descent

You're ready for:

- This week's quizzes
- Programming exercises
- Week 4 material on deep neural networks

## Key Takeaways

1. **Zero initialization fails**: All hidden units become identical (symmetry problem)
2. **Random initialization required**: Breaks symmetry so units learn different features
3. **Biases can be zero**: Only weights need to be random
4. **Use small values**: Multiply by `0.01` to avoid saturation
5. **Saturation problem**: Large weights → Large $Z$ → Small gradients → Slow learning
6. **Proof by induction**: Symmetry persists forever with zero initialization
7. **No redundant units**: Random init ensures all hidden units contribute
8. **Shallow vs deep**: Deep networks may need different initialization constants
9. **Xavier/He initialization**: Better strategies for deep networks (coming later)
10. **Always verify dimensions**: Check parameter shapes match architecture
11. **Logistic regression exception**: Zero init works for logistic regression (no hidden layers)
12. **ReLU less sensitive**: But small initialization still recommended
13. **Binary classification**: Especially important to keep weights small for output sigmoid
14. **Gaussian distribution**: `np.random.randn()` samples from $\mathcal{N}(0, 1)$
15. **Feature diversity**: The whole point of multiple hidden units is learning different features!
