---
title: Week 4 - Deep Neural Networks
parent: Neural Networks and Deep Learning
has_children: true
nav_order: 4
last_modified_date: 2025-11-23 10:00:00 +1100
---

# Week 4 - Deep Neural Networks
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Welcome to Week 4! This week, we'll extend everything you've learned about shallow neural networks (with one hidden layer) to **deep neural networks** with many layers. Deep networks are the foundation of modern deep learning and enable learning hierarchical representations of complex patterns.

## What You'll Learn

### Deep Network Fundamentals

- **Deep L-layer neural network architecture**: Understanding networks with arbitrary depth
- **Forward propagation in deep networks**: Computing predictions through multiple layers
- **Matrix dimensions in deep networks**: Keeping track of dimensions across many layers
- **Why deep networks work**: Understanding hierarchical feature learning

### Building Blocks

- **Forward and backward functions**: Modular implementation approach
- **Parameter initialization**: Proper initialization for deep networks
- **Hyperparameters vs parameters**: Understanding what to tune

### Practical Implementation

- **Building a deep neural network step-by-step**: From scratch implementation
- **Getting matrix dimensions right**: Debugging dimension mismatches
- **Circuit theory and deep learning**: Theoretical motivation for depth

## Key Concepts

### Deep vs Shallow

**Shallow neural network** (Weeks 1-3):

- Input layer + 1 hidden layer + output layer
- Limited representation capacity
- Good for simple problems

**Deep neural network** (This week):

- Input layer + L-1 hidden layers + output layer
- Can learn hierarchical features
- Essential for complex problems (vision, speech, NLP)

### Hierarchical Feature Learning

Deep networks learn features at multiple levels of abstraction:

**Example: Face recognition**

$$\text{Pixels} \xrightarrow{\text{Layer 1}} \text{Edges} \xrightarrow{\text{Layer 2}} \text{Face parts} \xrightarrow{\text{Layer 3}} \text{Faces}$$

Each layer builds on representations from the previous layer!

### Notation for Deep Networks

We'll use consistent notation throughout:

| Symbol | Meaning |
|--------|---------|
| $L$ | Number of layers (excluding input) |
| $n^{[l]}$ | Number of units in layer $l$ |
| $a^{[l]}$ | Activations in layer $l$ |
| $W^{[l]}, b^{[l]}$ | Parameters for layer $l$ |
| $Z^{[l]}$ | Linear output of layer $l$ |

**Example**: A 4-layer network has $L = 4$ with layers indexed $l = 1, 2, 3, 4$.

## Prerequisites

Before starting this week, make sure you're comfortable with:

- ✅ Neural network representation and notation
- ✅ Forward propagation for 2-layer networks
- ✅ Backpropagation and gradient computation
- ✅ Vectorization across training examples
- ✅ Activation functions and their derivatives
- ✅ Gradient descent optimization

If you need to review, see [Week 3 - Shallow Neural Networks]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/index.md %}).

## What Makes Deep Networks Powerful?

### 1. Hierarchical Representation

Deep networks learn features at multiple levels:

**Audio/Speech**:

- Layer 1: Basic sound waves
- Layer 2: Phonemes
- Layer 3: Words
- Layer 4: Sentences

**Images**:

- Layer 1: Edges and textures
- Layer 2: Simple shapes
- Layer 3: Object parts
- Layer 4: Objects

### 2. Computational Efficiency

**Circuit theory insight**: Some functions can be computed with exponentially fewer units using deep networks vs shallow networks!

$$\text{Shallow: } O(2^n) \text{ units needed}$$

$$\text{Deep: } O(\log n) \text{ layers needed}$$

### 3. Empirical Success

Deep networks have achieved breakthrough results in:

- Computer vision (ImageNet)
- Speech recognition (human-level accuracy)
- Natural language processing (GPT, BERT)
- Game playing (AlphaGo, AlphaZero)

## Learning Objectives

By the end of this week, you'll be able to:

1. **Describe** deep neural network architecture with $L$ layers
2. **Implement** forward propagation for deep networks
3. **Compute** gradients using backpropagation in deep networks
4. **Build** a deep neural network from scratch
5. **Initialize** parameters appropriately for deep networks
6. **Debug** dimension mismatches in deep networks
7. **Explain** why deep networks outperform shallow networks
8. **Identify** hyperparameters vs parameters
9. **Apply** deep networks to classification problems

## Why This Matters

Understanding deep networks is **essential** for modern deep learning:

- Most state-of-the-art models are deep (10-1000+ layers)
- Transfer learning relies on deep pretrained networks
- Deep architectures are the foundation of CNNs, RNNs, Transformers
- Industry applications almost always use deep networks

## Tips for Success

1. **Master the notation**: Deep networks have more indices - keep track carefully!
2. **Check dimensions frequently**: Most bugs are dimension mismatches
3. **Implement modularly**: Build reusable forward/backward functions
4. **Visualize the architecture**: Draw diagrams to understand data flow
5. **Start simple**: Test with small networks (L=2) before going deep
6. **Use vectorization**: Essential for performance with many layers

## Course Progress

You're now entering the final week of Course 1! After completing this week:

✅ Week 1: Introduction to Deep Learning  
✅ Week 2: Neural Networks Basics (Logistic Regression)  
✅ Week 3: Shallow Neural Networks (1 hidden layer)  
🔄 **Week 4: Deep Neural Networks (L layers)** ← You are here

Let's dive into deep neural networks! 🚀
