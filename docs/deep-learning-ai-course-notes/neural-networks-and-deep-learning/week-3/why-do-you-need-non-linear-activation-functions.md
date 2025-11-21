---
title: Why Do You Need Non-Linear Activation Functions?
parent: Week 3 - Shallow Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 7
last_modified_date: 2025-11-22 09:59:00 +1100
---

# Why Do You Need Non-Linear Activation Functions?
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

We've learned about various [activation functions]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/activation-functions.md %}) like ReLU, tanh, and sigmoid. But why do we need them at all? Why not just use linear functions (or no activation function)? Let's explore why **non-linear activation functions are essential** for neural networks to work.

## The Experiment: What if We Use Linear Activation?

![Two-layer neural network diagram showing three input nodes (x1, x2, x3) on the left connected to three hidden layer nodes in the middle, which connect to a single output node (ŷ) on the right. All connections between layers are shown with arrows, illustrating a fully connected feedforward architecture. The diagram demonstrates the basic structure used to explain why linear activation functions would collapse this multi-layer network into simple linear regression.](/assets/images/deep-learning/neural-networks/week-3/feedforward_architecture.png)

### Linear (Identity) Activation Function

Consider using a **linear activation function**:

$$g(z) = z$$

This is also called the **identity activation function** because it simply outputs whatever is input.

Let's apply this to our 2-layer neural network:

$$z^{[1]} = W^{[1]} x + b^{[1]}$$

$$a^{[1]} = g(z^{[1]}) = z^{[1]} \quad \text{(linear activation)}$$

$$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$$

$$a^{[2]} = g(z^{[2]}) = z^{[2]} \quad \text{(linear activation)}$$

$$\hat{y} = a^{[2]}$$

Seems reasonable, right? Let's see what actually happens...

## The Problem: Collapsing to Linear Regression

### Mathematical Proof

Let's substitute $a^{[1]} = z^{[1]}$ into the equation for $z^{[2]}$:

**Step 1:** Hidden layer output

$$a^{[1]} = z^{[1]} = W^{[1]} x + b^{[1]}$$

**Step 2:** Output layer computation

$$a^{[2]} = z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$$

**Step 3:** Substitute $a^{[1]}$ from Step 1

$$a^{[2]} = W^{[2]} (W^{[1]} x + b^{[1]}) + b^{[2]}$$

**Step 4:** Distribute $W^{[2]}$

$$a^{[2]} = W^{[2]} W^{[1]} x + W^{[2]} b^{[1]} + b^{[2]}$$

**Step 5:** Simplify by defining new parameters

Let:

- $W' = W^{[2]} W^{[1]}$ (matrix multiplication)
- $b' = W^{[2]} b^{[1]} + b^{[2]}$

Then:

$$a^{[2]} = W' x + b'$$

### The Devastating Result

$$\hat{y} = W' x + b'$$

This is just **linear regression**! The neural network with a hidden layer is computing exactly the same function as a model with **no hidden layers at all**.

> **Key insight**: The composition of two linear functions is itself a linear function. No matter how many hidden layers you add, if all activations are linear, the entire network is equivalent to a single linear transformation.

## What About Deep Networks?

### Many Layers, Same Problem

Even with 10, 100, or 1000 hidden layers, if all use linear activation:

$$a^{[L]} = W^{[L]} W^{[L-1]} \cdots W^{[2]} W^{[1]} x + b'$$

This simplifies to:

$$a^{[L]} = W_{\text{combined}} x + b_{\text{combined}}$$

**Result**: Still just linear regression, regardless of depth!

### Hidden Layers Become Useless

With linear activations:

- Adding hidden layers provides **no benefit**
- The network cannot learn complex patterns
- You might as well use simple linear regression

## Mixed Activation Functions

### Linear Hidden + Sigmoid Output

What if we use linear activation in hidden layers but sigmoid at the output?

$$a^{[1]} = W^{[1]} x + b^{[1]} \quad \text{(linear)}$$

$$a^{[2]} = \sigma(W^{[2]} a^{[1]} + b^{[2]}) = \sigma(W' x + b')$$

**Result**: This is just **logistic regression** without any hidden layer! The model is no more expressive than standard logistic regression.

## Why Non-Linearity is Essential

### Breaking the Linear Composition

Non-linear activation functions like ReLU, tanh, or sigmoid **break the chain** of linear compositions:

$$a^{[1]} = \text{ReLU}(W^{[1]} x + b^{[1]})$$

$$a^{[2]} = \text{ReLU}(W^{[2]} a^{[1]} + b^{[2]})$$

Now the composition is **non-linear**, allowing the network to:

- Learn complex decision boundaries
- Approximate any function (universal approximation theorem)
- Extract hierarchical features
- Benefit from depth

### Visual Intuition

```
Linear Network (useless):
Input → Linear → Linear → Linear → Output
  x  →   Wx+b →  Wx+b →  Wx+b → W'x+b'  (collapsed!)

Non-Linear Network (powerful):
Input → ReLU → ReLU → ReLU → Output
  x  →  max(0,Wx+b) → (non-linear) → complex function
```

## The One Exception: Regression Output Layer

### When Linear Activation is Acceptable

There is **one place** where linear activation makes sense: the **output layer for regression problems**.

**Scenario**: Predicting continuous real values (e.g., housing prices)

$$y \in (-\infty, \infty) \quad \text{or} \quad y \in [0, \infty)$$

**Example Architecture**:

```python
# Hidden layers: Use non-linear activations
Z1 = np.dot(W1, X) + b1
A1 = relu(Z1)                    # Non-linear!

Z2 = np.dot(W2, A1) + b2
A2 = relu(Z2)                    # Non-linear!

# Output layer: Linear activation for regression
Z3 = np.dot(W3, A2) + b3
A3 = Z3                          # Linear (or identity)
```

**Why this works**:

- Hidden layers use **non-linear** activations (ReLU, tanh, etc.)
- They extract complex features
- Output layer uses **linear** activation to produce any real number

### Alternative for Non-Negative Outputs

For predictions that should be non-negative (e.g., housing prices ≥ 0):

```python
# Output layer: ReLU for non-negative values
Z3 = np.dot(W3, A2) + b3
A3 = relu(Z3)                    # Ensures y_hat ≥ 0
```

## Summary of Rules

### Hidden Layers

| Activation | Use in Hidden Layers? |
|------------|-----------------------|
| Linear (identity) | ❌ **Almost never** (except rare compression cases) |
| ReLU | ✅ **Default choice** |
| Leaky ReLU | ✅ Good alternative |
| tanh | ✅ Sometimes useful |
| Sigmoid | ❌ Rarely recommended |

### Output Layer

| Problem Type | Output Activation |
|--------------|-------------------|
| Binary classification | Sigmoid |
| Multi-class classification | Softmax |
| Regression (any real value) | Linear (identity) |
| Regression (non-negative) | ReLU or Linear |

## Complete Example: Housing Price Prediction

### Problem Setup

- **Input**: Features like square footage, bedrooms, location
- **Output**: Price (continuous, non-negative)

### Correct Architecture

```python
# Layer 1: Non-linear activation
Z1 = np.dot(W1, X) + b1
A1 = relu(Z1)                    # ✅ Non-linear

# Layer 2: Non-linear activation  
Z2 = np.dot(W2, A1) + b2
A2 = relu(Z2)                    # ✅ Non-linear

# Output layer: ReLU for non-negative prices
Z3 = np.dot(W3, A2) + b3
A3 = relu(Z3)                    # ✅ Ensures price ≥ 0

y_hat = A3
```

### Wrong Architecture (Don't Do This!)

```python
# Layer 1: Linear activation
Z1 = np.dot(W1, X) + b1
A1 = Z1                          # ❌ Linear (useless!)

# Layer 2: Linear activation
Z2 = np.dot(W2, A1) + b2
A2 = Z2                          # ❌ Linear (useless!)

# Output layer: Linear
Z3 = np.dot(W3, A2) + b3
A3 = Z3                          # ❌ Entire network collapses to W'x + b'

y_hat = A3  # This is just linear regression!
```

## Key Takeaways

1. **Linear activations are useless in hidden layers** - they collapse the network to linear regression
2. **Composition of linear functions is linear** - $W^{[2]}(W^{[1]}x + b^{[1]}) + b^{[2]} = W'x + b'$
3. **Non-linearity is essential** for neural networks to learn complex functions
4. **Hidden layers must use non-linear activations** (ReLU, tanh, Leaky ReLU, etc.)
5. **Output layer can use linear activation** for regression problems with unbounded outputs
6. **Even with 1000 layers**, all-linear activation = simple linear regression
7. **Breaking the linear chain** with non-linearity enables deep learning's power
8. **Universal approximation theorem** requires non-linear activations to work
9. **Use ReLU for non-negative regression outputs** (e.g., prices, quantities)
10. **Exception is rare**: Linear activations in hidden layers are only used in special compression scenarios
