---
layout: post
title: "What is a Neural Network?"
date: 2025-11-16 00:00:00 +0000
categories: [Deep Learning, Course Notes, Neural Networks and Deep Learning, Week 1 Introduction to Deep Learning]
tags: [neural-networks, deep-learning, ai]
---

## Introduction

A neural network is a computational model inspired by the way biological neural networks in the human brain process information. In its simplest form, a neural network takes inputs, processes them through hidden layers, and produces outputs.

## Basic Structure

### Single Neuron

The most basic unit of a neural network is a single neuron (also called a perceptron). It performs a simple computation:

$$
y = f(w^T x + b)
$$

Where:
- $x$ is the input vector
- $w$ is the weight vector
- $b$ is the bias term
- $f$ is the activation function
- $y$ is the output

### Example: Housing Price Prediction

Let's consider a simple example of predicting housing prices based on size:

Given:
- Input: Size of the house (in square feet)
- Output: Price

The neuron learns a function that maps size to price:

$$
\text{price} = \max(0, w \cdot \text{size} + b)
$$

Here we use the ReLU (Rectified Linear Unit) activation function: $f(z) = \max(0, z)$

### Multi-Layer Neural Network

A more complex neural network consists of:
1. **Input Layer**: Receives the input features
2. **Hidden Layers**: Process the inputs through multiple transformations
3. **Output Layer**: Produces the final prediction

For a network with one hidden layer:

$$
\begin{aligned}
z^{[1]} &= W^{[1]} x + b^{[1]} \\
a^{[1]} &= f(z^{[1]}) \\
z^{[2]} &= W^{[2]} a^{[1]} + b^{[2]} \\
y &= f(z^{[2]})
\end{aligned}
$$

Where:
- Superscript $[i]$ denotes the layer number
- $W^{[i]}$ are the weight matrices
- $b^{[i]}$ are the bias vectors
- $a^{[i]}$ are the activations

## Common Activation Functions

### ReLU (Rectified Linear Unit)
$$
f(z) = \max(0, z)
$$

### Sigmoid
$$
f(z) = \frac{1}{1 + e^{-z}}
$$

### Tanh (Hyperbolic Tangent)
$$
f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

## Why Deep Learning?

Deep neural networks with multiple hidden layers can learn hierarchical representations:

1. **First layers** might learn simple features (edges, colors)
2. **Middle layers** combine simple features into more complex patterns
3. **Deep layers** recognize high-level concepts

## Mathematical Notation

In deep learning, we often work with:

- **Vectors**: $x \in \mathbb{R}^n$ (lowercase, bold)
- **Matrices**: $W \in \mathbb{R}^{m \times n}$ (uppercase, bold)
- **Scalars**: $b \in \mathbb{R}$ (regular font)

### Forward Propagation

The forward propagation for a sample $x$ through layer $l$ can be expressed as:

$$
a^{[l]} = f^{[l]}(W^{[l]} a^{[l-1]} + b^{[l]})
$$

Where $a^{[0]} = x$ (the input).

## Example: XOR Problem

Consider the XOR function, which cannot be solved by a single neuron but can be solved with a two-layer network.

Truth table:
| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

A two-layer network can learn this function by creating a non-linear decision boundary.

## Key Takeaways

1. Neural networks are composed of layers of interconnected neurons
2. Each neuron performs a weighted sum followed by a non-linear activation
3. Deep networks can learn complex, hierarchical representations
4. The power comes from the composition of simple operations through multiple layers

## Code Example

```python
import numpy as np

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, W1, b1, W2, b2):
    """
    Forward propagation for a 2-layer neural network
    
    Args:
        X: Input data of shape (n_features, m_samples)
        W1: Weight matrix for layer 1
        b1: Bias vector for layer 1
        W2: Weight matrix for layer 2
        b2: Bias vector for layer 2
    
    Returns:
        A2: Output predictions
    """
    # Layer 1
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)  # ReLU activation
    
    # Layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # Sigmoid activation
    
    return A2
```

## Next Steps

In the following lessons, we'll explore:
- How to train neural networks (backpropagation)
- Optimization algorithms (gradient descent, Adam, etc.)
- Regularization techniques
- Practical tips for building deep learning systems

---

*This is part of the "Neural Networks and Deep Learning" course notes.*
