---
title: Neural Network Representation
parent: Week 3 - Shallow Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 2
last_modified_date: 2025-11-22 09:18:00 +1100
---

# Neural Network Representation
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

In this lesson, we'll explore the structure and notation of neural networks, focusing on a simple architecture with one hidden layer (called a **2-layer neural network**). Understanding how to properly represent and describe neural networks is fundamental for implementing them.

## Anatomy of a Neural Network

A neural network consists of organized layers of nodes (neurons). Let's break down the components:

### Three Types of Layers

Consider a neural network with 3 input features and 4 hidden units:

![A 2-layer neural network diagram showing three input nodes (x1, x2, x3) on the left as the input layer, four hidden nodes in the middle connected by arrows representing the hidden layer, and one output node on the right producing prediction ŷ. Handwritten annotations label each layer: input layer with a^[0] equals X, hidden layer with four activation nodes a1^[1] through a4^[1], and output layer with a^[2] equals ŷ. Additional notes show the weight matrices W^[1] and W^[2], bias vectors b^[1] and b^[2], and indicate this is a 2 layer neural network. All nodes are connected with directed edges showing forward propagation flow from left to right. Mathematical notation on the right shows the activation vector a^[1] containing four elements and emphasizes the layer indexing convention using square brackets.](/assets/images/deep-learning/neural-networks/week-3/2_layer_neural_network_visual.png)

1. **Input Layer (Layer 0)**
   - Contains the input features: $x_1, x_2, x_3$
   - Stacked vertically as a column vector
   - Denoted as $a^{[0]} = X$
   - Not counted when describing network depth

2. **Hidden Layer (Layer 1)**
   - Contains intermediate computations
   - In this example: 4 nodes (hidden units)
   - Called "hidden" because values aren't directly observed in training data
   - We only see inputs ($X$) and outputs ($y$), not these intermediate values

3. **Output Layer (Layer 2)**
   - Produces the final prediction $\hat{y}$
   - In this example: 1 node for binary classification
   - Generates the predicted value

> **Note**: This is called a **2-layer neural network** because we don't count the input layer. The convention counts only layers with learnable parameters (hidden + output layers).

## Notation for Activations

The term **activations** refers to the values that each layer computes and passes to the next layer.

### Layer 0 (Input Layer)

$$a^{[0]} = X = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$$

The input features can be denoted as either $X$ or $a^{[0]}$.

### Layer 1 (Hidden Layer)

Each node in the hidden layer computes an activation:

$$a^{[1]} = \begin{bmatrix} a^{[1]}_1 \\ a^{[1]}_2 \\ a^{[1]}_3 \\ a^{[1]}_4 \end{bmatrix}$$

where:

- $a^{[1]}_1$ = activation of first hidden unit
- $a^{[1]}_2$ = activation of second hidden unit
- $a^{[1]}_3$ = activation of third hidden unit
- $a^{[1]}_4$ = activation of fourth hidden unit

In Python (NumPy), this is a $(4, 1)$ vector (4 rows, 1 column).

### Layer 2 (Output Layer)

$$a^{[2]} = \hat{y}$$

The output layer produces a single real number (for binary classification), which is our prediction.

## Notation Conventions

### Superscript Notation Summary

| Notation | Meaning | Example |
|----------|---------|---------|
| $a^{[l]}$ | Activations from layer $l$ | $a^{[1]}$ = hidden layer activations |
| $a^{(i)}$ | Value for training example $i$ | $x^{(i)}$ = features of example $i$ |
| $a^{[l]}_j$ | Activation of unit $j$ in layer $l$ | $a^{[1]}_3$ = 3rd hidden unit |

**Key distinction:**

- **Square brackets** $[l]$: layer number
- **Round brackets** $(i)$: training example number
- **Subscript** $j$: unit/node number within a layer

### Why Logistic Regression Had No Brackets

In [logistic regression]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/logistic-regression.md %}), we used $\hat{y} = a$ without superscripts because there was only one output layer. With neural networks, we need the $[l]$ notation to distinguish between layers.

## Parameters for Each Layer

Each layer with computations (hidden and output) has associated parameters:

### Hidden Layer (Layer 1) Parameters

- $W^{[1]}$: weight matrix, shape $(4, 3)$
  - 4 rows: one for each hidden unit
  - 3 columns: one for each input feature
- $b^{[1]}$: bias vector, shape $(4, 1)$
  - 4 rows: one bias for each hidden unit

### Output Layer (Layer 2) Parameters

- $W^{[2]}$: weight matrix, shape $(1, 4)$
  - 1 row: one output unit
  - 4 columns: one for each hidden unit
- $b^{[2]}$: bias scalar, shape $(1, 1)$
  - Single bias for the output

### Parameter Dimensions Summary

For a network with $n_x$ input features, $n^{[1]}$ hidden units, and $n^{[2]}$ output units:

| Parameter | Dimensions | Example |
|-----------|------------|---------|
| $W^{[1]}$ | $(n^{[1]}, n_x)$ | $(4, 3)$ |
| $b^{[1]}$ | $(n^{[1]}, 1)$ | $(4, 1)$ |
| $W^{[2]}$ | $(n^{[2]}, n^{[1]})$ | $(1, 4)$ |
| $b^{[2]}$ | $(n^{[2]}, 1)$ | $(1, 1)$ |

We'll explore these dimensions in more detail when implementing forward propagation.

## What's Next

Now that we understand the structure and notation of a neural network, the next step is to understand the computations: how does the network transform inputs $X$ all the way through to predictions $\hat{y}$? We'll cover this in the next lesson on computing neural network output.

## Key Takeaways

1. A **2-layer neural network** has one hidden layer (input layer not counted)
2. **Hidden layers** are called "hidden" because their true values aren't observed in training data
3. **Activations** $a^{[l]}$ are the outputs computed by layer $l$
4. **Notation conventions**:
   - $[l]$ for layer number
   - $(i)$ for training example
   - $j$ subscript for unit within a layer
5. Each layer has parameters $W^{[l]}$ (weights) and $b^{[l]}$ (bias)
6. Parameter dimensions depend on the number of units in current and previous layers
