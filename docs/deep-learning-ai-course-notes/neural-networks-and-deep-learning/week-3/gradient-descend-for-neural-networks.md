---
title: Gradient Descent for Neural Networks
parent: Week 3 - Shallow Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 9
last_modified_date: 2025-11-23 09:45:00 +1100
---

# Gradient Descent for Neural Networks
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

This lesson provides the complete equations needed to implement **gradient descent** for a neural network with one hidden layer. You'll learn the practical formulas for both **forward propagation** and **backpropagation**, which together enable you to train your neural network.

## Network Architecture

We'll work with a **two-layer neural network** (one hidden layer):

### Parameters

For a network with:

- $n_x = n^{[0]}$ input features
- $n^{[1]}$ hidden units
- $n^{[2]}$ output units (typically $n^{[2]} = 1$ for binary classification)

The parameters have the following dimensions:

| Parameter | Dimensions | Description |
|-----------|------------|-------------|
| $W^{[1]}$ | $(n^{[1]}, n^{[0]})$ | Weights from input to hidden layer |
| $b^{[1]}$ | $(n^{[1]}, 1)$ | Biases for hidden layer |
| $W^{[2]}$ | $(n^{[2]}, n^{[1]})$ | Weights from hidden to output layer |
| $b^{[2]}$ | $(n^{[2]}, 1)$ | Biases for output layer |

### Cost Function

For **binary classification**, we use the logistic regression loss:

$$J(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

where:

$$\mathcal{L}(\hat{y}, y) = -\left( y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right)$$

and $\hat{y} = a^{[2]}$ is the network's prediction.

## Gradient Descent Algorithm

### Overview

The training process follows these steps:

1. **Initialize** parameters randomly (not zeros!)
2. **Repeat** until convergence:
   - **Forward propagation**: Compute predictions $\hat{y}^{(i)}$ for all examples
   - **Compute cost**: Evaluate $J$
   - **Backpropagation**: Compute gradients $dW^{[1]}$, $db^{[1]}$, $dW^{[2]}$, $db^{[2]}$
   - **Update parameters**: Apply gradient descent

### Parameter Updates

In each iteration, update all parameters:

$$W^{[1]} := W^{[1]} - \alpha \, dW^{[1]}$$

$$b^{[1]} := b^{[1]} - \alpha \, db^{[1]}$$

$$W^{[2]} := W^{[2]} - \alpha \, dW^{[2]}$$

$$b^{[2]} := b^{[2]} - \alpha \, db^{[2]}$$

where $\alpha$ is the learning rate.

## Forward Propagation Equations

Given training data $X$ (shape: $(n^{[0]}, m)$), compute activations for all $m$ examples:

### Layer 1 (Hidden Layer)

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

$$A^{[1]} = g^{[1]}(Z^{[1]})$$

where $g^{[1]}$ is the activation function (e.g., tanh, ReLU).

### Layer 2 (Output Layer)

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

$$A^{[2]} = g^{[2]}(Z^{[2]})$$

For binary classification: $g^{[2]} = \sigma$ (sigmoid function).

### Summary

**Forward propagation in 4 equations:**

$$\boxed{\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]} \\
A^{[1]} &= g^{[1]}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} &= \sigma(Z^{[2]})
\end{align}}$$

## Backpropagation Equations

Now compute the gradients needed for parameter updates. All computations are **vectorized** across all $m$ training examples.

### Output Layer Gradients

**Step 1**: Compute error at output layer

$$dZ^{[2]} = A^{[2]} - Y$$

where $Y$ is the $(1, m)$ matrix of true labels: $Y = [y^{(1)}, y^{(2)}, \ldots, y^{(m)}]$

> **Note**: For binary classification with sigmoid output, this formula $dZ^{[2]} = A^{[2]} - Y$ combines both the derivative of the cost and the sigmoid activation.

**Step 2**: Compute weight gradient

$$dW^{[2]} = \frac{1}{m} dZ^{[2]} (A^{[1]})^T$$

**Step 3**: Compute bias gradient

$$db^{[2]} = \frac{1}{m} \text{np.sum}(dZ^{[2]}, \text{axis}=1, \text{keepdims}=\text{True})$$

### Hidden Layer Gradients

**Step 4**: Backpropagate error to hidden layer

$$dZ^{[1]} = (W^{[2]})^T dZ^{[2]} \odot g^{[1]'}(Z^{[1]})$$

where:
- $\odot$ denotes element-wise multiplication
- $g^{[1]'}(Z^{[1]})$ is the [derivative of the activation function]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/derivatives-of-activation-functions.md %})

**Step 5**: Compute weight gradient

$$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$$

**Step 6**: Compute bias gradient

$$db^{[1]} = \frac{1}{m} \text{np.sum}(dZ^{[1]}, \text{axis}=1, \text{keepdims}=\text{True})$$

### Summary

**Backpropagation in 6 equations:**

$$\boxed{\begin{align}
dZ^{[2]} &= A^{[2]} - Y \\
dW^{[2]} &= \frac{1}{m} dZ^{[2]} (A^{[1]})^T \\
db^{[2]} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[2](i)} \\
dZ^{[1]} &= (W^{[2]})^T dZ^{[2]} \odot g^{[1]'}(Z^{[1]}) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} X^T \\
db^{[1]} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[1](i)}
\end{align}}$$

## Matrix Dimensions

Understanding dimensions helps debug implementation errors:

### Forward Propagation

| Variable | Dimensions | Description |
|----------|------------|-------------|
| $X$ | $(n^{[0]}, m)$ | Input features |
| $Z^{[1]}$ | $(n^{[1]}, m)$ | Hidden layer linear output |
| $A^{[1]}$ | $(n^{[1]}, m)$ | Hidden layer activations |
| $Z^{[2]}$ | $(n^{[2]}, m)$ | Output layer linear output |
| $A^{[2]}$ | $(n^{[2]}, m)$ | Output predictions |

### Backpropagation

| Variable | Dimensions | Description |
|----------|------------|-------------|
| $dZ^{[2]}$ | $(n^{[2]}, m)$ | Output layer error |
| $dW^{[2]}$ | $(n^{[2]}, n^{[1]})$ | Output weights gradient (same as $W^{[2]}$) |
| $db^{[2]}$ | $(n^{[2]}, 1)$ | Output bias gradient (same as $b^{[2]}$) |
| $dZ^{[1]}$ | $(n^{[1]}, m)$ | Hidden layer error |
| $dW^{[1]}$ | $(n^{[1]}, n^{[0]})$ | Hidden weights gradient (same as $W^{[1]}$) |
| $db^{[1]}$ | $(n^{[1]}, 1)$ | Hidden bias gradient (same as $b^{[1]}$) |

> **Pattern**: Gradient dimensions match their corresponding parameter dimensions!

## Implementation Tips

### The `keepdims` Parameter

When computing bias gradients with `np.sum`, use `keepdims=True`:

```python
db = np.sum(dZ, axis=1, keepdims=True)
```

**Why?**
- **Without `keepdims=True`**: NumPy outputs a "rank-1 array" with shape `(n,)` instead of `(n, 1)`
- **With `keepdims=True`**: Output is a proper column vector `(n, 1)`

This prevents dimension mismatches in subsequent calculations!

**Example:**

```python
# Without keepdims - problematic
dZ = np.array([[1, 2, 3],
               [4, 5, 6]])  # Shape: (2, 3)

db_bad = np.sum(dZ, axis=1)  # Shape: (2,) - rank-1 array
print(db_bad.shape)  # (2,)

# With keepdims - correct
db_good = np.sum(dZ, axis=1, keepdims=True)  # Shape: (2, 1)
print(db_good.shape)  # (2, 1)
```

**Alternative**: Explicitly reshape after summation:

```python
db = np.sum(dZ, axis=1).reshape((n, 1))
```

### Random Initialization

> **Critical**: Always initialize weights **randomly**, not to zeros!

```python
# Good - random initialization
W1 = np.random.randn(n1, n0) * 0.01
b1 = np.zeros((n1, 1))  # Biases can be zero

W2 = np.random.randn(n2, n1) * 0.01
b2 = np.zeros((n2, 1))

# Bad - all zeros (will fail!)
W1 = np.zeros((n1, n0))  # ❌ Don't do this!
```

**Why?** We'll cover this in a later lesson, but briefly: zero initialization causes all neurons to learn identical features (symmetry problem).

## Complete Implementation

```python
import numpy as np

def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize parameters randomly

    Args:
        n_x: input layer size
        n_h: hidden layer size
        n_y: output layer size

    Returns:
        parameters: dictionary containing W1, b1, W2, b2
    """
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

def forward_propagation(X, parameters):
    """
    Implement forward propagation

    Args:
        X: input data of shape (n_x, m)
        parameters: dictionary containing W1, b1, W2, b2

    Returns:
        A2: predictions of shape (1, m)
        cache: dictionary containing Z1, A1, Z2, A2 for backpropagation
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Forward propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # or use ReLU
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # for binary classification

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

def compute_cost(A2, Y):
    """
    Compute cross-entropy cost

    Args:
        A2: predictions of shape (1, m)
        Y: true labels of shape (1, m)

    Returns:
        cost: cross-entropy cost
    """
    m = Y.shape[1]

    # Compute cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))
    cost = -np.sum(logprobs) / m

    return cost

def backward_propagation(X, Y, parameters, cache):
    """
    Implement backpropagation

    Args:
        X: input data of shape (n_x, m)
        Y: true labels of shape (1, m)
        parameters: dictionary containing W1, b1, W2, b2
        cache: dictionary containing Z1, A1, Z2, A2

    Returns:
        grads: dictionary containing dW1, db1, dW2, db2
    """
    m = X.shape[1]

    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    # Backpropagation
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # tanh derivative
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Args:
        parameters: dictionary containing W1, b1, W2, b2
        grads: dictionary containing dW1, db1, dW2, db2
        learning_rate: learning rate for gradient descent

    Returns:
        parameters: updated parameters
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update parameters
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))
```

## Training Loop Example

```python
# Training a neural network
def train_neural_network(X, Y, n_h, learning_rate=0.01, num_iterations=10000):
    """
    Train a 2-layer neural network

    Args:
        X: training data (n_x, m)
        Y: labels (1, m)
        n_h: hidden layer size
        learning_rate: learning rate for gradient descent
        num_iterations: number of training iterations

    Returns:
        parameters: trained parameters
    """
    n_x = X.shape[0]  # input size
    n_y = Y.shape[0]  # output size

    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Training loop
    for i in range(num_iterations):
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(A2, Y)

        # Backpropagation
        grads = backward_propagation(X, Y, parameters, cache)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print cost every 1000 iterations
        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")

    return parameters
```

## Similarity to Logistic Regression

Notice that the **output layer** equations are identical to logistic regression:

$$dZ^{[2]} = A^{[2]} - Y$$

$$dW^{[2]} = \frac{1}{m} dZ^{[2]} (A^{[1]})^T$$

$$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

**Why?** The output layer *is* essentially logistic regression, but it operates on the learned features $A^{[1]}$ from the hidden layer instead of raw inputs $X$!

## Do You Need to Understand the Calculus?

**Short answer**: Not necessarily!

Many successful deep learning practitioners implement these equations without deeply understanding the calculus derivations. As long as you:
1. Implement the equations correctly
2. Verify dimensions match
3. Test your implementation

You can build effective neural networks.

**However**, understanding the derivations helps with:
- Debugging when things go wrong
- Designing new architectures
- Intuition about how changes affect training

## Key Takeaways

1. **Parameters**: Neural network has 4 parameter matrices: $W^{[1]}$, $b^{[1]}$, $W^{[2]}$, $b^{[2]}$
2. **Forward propagation**: 4 equations compute predictions from inputs
3. **Backpropagation**: 6 equations compute gradients for all parameters
4. **Vectorization**: All equations work on entire training set simultaneously (shape: `(..., m)`)
5. **Gradient descent**: Update rule is $\theta := \theta - \alpha \, d\theta$
6. **Dimensions matter**: Gradients have same dimensions as their parameters
7. **`keepdims=True`**: Essential for maintaining proper matrix dimensions in NumPy
8. **Random initialization**: Critical for breaking symmetry (never initialize to zeros!)
9. **Output layer**: Similar to logistic regression (operates on learned features)
10. **Calculus optional**: Can implement successfully without deep mathematical understanding
11. **Element-wise multiplication**: $\odot$ operator for $dZ^{[1]}$ computation
12. **Activation derivatives**: Need $g'(Z^{[1]})$ from activation function (see [derivatives lesson]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/derivatives-of-activation-functions.md %}))
13. **Pattern recognition**: Forward prop computes $Z$, $A$; backprop computes $dZ$, $dW$, $db$
14. **Cost function**: Cross-entropy loss for binary classification
15. **Training loop**: Initialize → Forward → Cost → Backward → Update → Repeat
