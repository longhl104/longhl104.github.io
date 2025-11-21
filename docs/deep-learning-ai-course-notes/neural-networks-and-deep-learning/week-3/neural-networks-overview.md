---
title: Neural Networks Overview
parent: Week 3 - Shallow Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 1
last_modified_date: 2025-11-21 13:26:00 +1100
---

# Neural Networks Overview
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

In the [previous week]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/index.md %}), we learned about logistic regression and how to vectorize it for efficiency. Now we'll extend these concepts to build **neural networks** with hidden layers.

**Goal**: Understand the basic structure and components of a neural network.

## From Logistic Regression to Neural Networks

### Logistic Regression Review

Recall logistic regression:

$$z = w^T x + b$$
$$a = \sigma(z)$$

**Components**:

- Input: $x$
- Parameters: $w$, $b$
- Linear combination: $z$
- Activation: $a = \sigma(z)$
- Output: $\hat{y} = a$

### What is a Neural Network?

A neural network is essentially **stacking multiple logistic regression units** together.

**Key idea**: Instead of going directly from input to output, we add **hidden layers** that learn intermediate representations.

## Neural Network Structure

![Neural network architecture diagram showing three representations: top shows a simple single-layer network with three inputs x1, x2, x3 connecting to output ŷ equals a; middle shows a two-layer network with three input nodes x1, x2, x3 connecting to hidden layer nodes marked with superscript [1], which connect to output ŷ equals a superscript [1]; bottom shows detailed computational flow with layers labeled, displaying forward propagation equations z[1] = W[1]x + b[1], a[1] = σ(z[1]), z[2] = W[2]a[1] + b[2], a[2] = σ(z[2]), and backward propagation with red arrows indicating derivatives dz, da flowing from loss function L(a,y) back through the network. Annotations in blue indicate parameters W[1], b[1], W[2], b[2] at each layer. Credit line shows Andrew Ng name at bottom right. The diagram illustrates progressive complexity from single neuron to multi-layer network architecture.](/assets/images/deep-learning/neural-networks/week-3/neural_network_architecture_diagram.png)

### Single Hidden Layer Network

A neural network with one hidden layer has three layers:

1. **Input Layer**: Contains the input features
2. **Hidden Layer**: Intermediate layer that learns representations
3. **Output Layer**: Produces the final prediction

### Example Architecture

**Input Layer** (Layer 0):

- Features: $x_1, x_2, x_3$

**Hidden Layer** (Layer 1):

- Nodes: $a^{[1]}_1, a^{[1]}_2, a^{[1]}_3, a^{[1]}_4$

**Output Layer** (Layer 2):

- Node: $a^{[2]} = \hat{y}$

### Notation Convention

**Superscript $[l]$**: Indicates layer number

- $a^{[1]}$ = activations in layer 1 (hidden layer)
- $a^{[2]}$ = activations in layer 2 (output layer)
- $W^{[1]}$, $b^{[1]}$ = parameters for layer 1
- $W^{[2]}$, $b^{[2]}$ = parameters for layer 2

**Superscript $(i)$**: Indicates training example number

- $x^{(i)}$ = $i$-th training example

**Subscript**: Indicates node/unit number

- $a^{[1]}_1$ = first node in layer 1
- $a^{[1]}_2$ = second node in layer 1

## How Each Node Works

Each node in a neural network performs two steps:

### Step 1: Linear Combination

$$z = w^T x + b$$

### Step 2: Activation Function

$$a = \sigma(z)$$

**For hidden layer node $j$**:

$$z^{[1]}_j = w^{[1]T}_j x + b^{[1]}_j$$
$$a^{[1]}_j = \sigma(z^{[1]}_j)$$

## Why Hidden Layers?

**Problem with logistic regression**: Can only learn linear decision boundaries.

**Solution with hidden layers**:

- Each hidden layer learns more complex features
- Combines input features in non-linear ways
- Enables learning of complex patterns

**Example**:

- Input: Raw pixel values
- Hidden layer: Learns edges, simple shapes
- Output: Classifies the image

## Neural Network Computation

### Forward Propagation

**Computing hidden layer**:

For each node $j$ in hidden layer:

$$z^{[1]}_j = w^{[1]T}_j x + b^{[1]}_j$$
$$a^{[1]}_j = \sigma(z^{[1]}_j)$$

**Computing output**:

$$z^{[2]} = w^{[2]T} a^{[1]} + b^{[2]}$$
$$a^{[2]} = \sigma(z^{[2]}) = \hat{y}$$

### Vectorized Computation

Instead of computing each node separately, we can vectorize:

**Hidden layer** (all nodes at once):

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[1]} = \sigma(Z^{[1]})$$

**Output layer**:

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = \sigma(Z^{[2]})$$

## Matrix Dimensions

Understanding dimensions is crucial for implementation.

### For One Training Example

**Input**: $x \in \mathbb{R}^{n_x}$ (column vector)

**Hidden layer**:

- $W^{[1]} \in \mathbb{R}^{n^{[1]} \times n_x}$ where $n^{[1]}$ = number of hidden units
- $b^{[1]} \in \mathbb{R}^{n^{[1]}}$
- $Z^{[1]} \in \mathbb{R}^{n^{[1]}}$
- $A^{[1]} \in \mathbb{R}^{n^{[1]}}$

**Output layer**:

- $W^{[2]} \in \mathbb{R}^{1 \times n^{[1]}}$ (for binary classification)
- $b^{[2]} \in \mathbb{R}$
- $Z^{[2]} \in \mathbb{R}$
- $A^{[2]} \in \mathbb{R}$

### For $m$ Training Examples

We stack examples horizontally (as columns):

**Input**: $X \in \mathbb{R}^{n_x \times m}$

**Hidden layer**:

- $Z^{[1]} \in \mathbb{R}^{n^{[1]} \times m}$
- $A^{[1]} \in \mathbb{R}^{n^{[1]} \times m}$

**Output layer**:

- $Z^{[2]} \in \mathbb{R}^{1 \times m}$
- $A^{[2]} \in \mathbb{R}^{1 \times m}$

## Comparison: Logistic Regression vs Neural Network

| Aspect | Logistic Regression | Neural Network |
|--------|-------------------|----------------|
| **Layers** | 2 (input, output) | 3+ (input, hidden(s), output) |
| **Parameters** | $w$, $b$ | $W^{[1]}$, $b^{[1]}$, $W^{[2]}$, $b^{[2]}$, ... |
| **Complexity** | Linear decision boundary | Non-linear decision boundaries |
| **Representations** | Uses raw features | Learns feature representations |
| **Computation** | Single step | Multiple steps (layers) |

## Key Takeaways

1. **Neural networks** stack multiple logistic regression-like units
2. **Hidden layers** learn intermediate representations
3. **Notation**: Use $[l]$ for layer number, $(i)$ for example number
4. **Each node**: Computes $z = w^T x + b$, then $a = \sigma(z)$
5. **Forward propagation**: Compute activations layer by layer
6. **Vectorization**: Process all examples and nodes simultaneously
7. **More layers**: Enable learning of more complex patterns
8. **Matrix dimensions**: Critical for correct implementation

**Remember**: A neural network is just organized logistic regression units that learn hierarchical representations!
