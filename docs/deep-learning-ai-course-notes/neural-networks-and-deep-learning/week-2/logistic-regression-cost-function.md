---
title: Logistic Regression Cost Function
parent: Week 2 - Neural Networks Basics
grand_parent: Neural Networks and Deep Learning
nav_order: 3
last_modified_date: 2025-11-19 00:21:00 +1100
---

# Logistic Regression Cost Function
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

To train a logistic regression model, we need to define a **cost function** that measures how well our model performs.

**Recap from previous lesson**:

$$\hat{y} = \sigma(w^T x + b)$$

where $\sigma(z)$ is the sigmoid function.

## Goal of Training

**Objective**: Find parameters $w$ and $b$ such that predictions $\hat{y}^{(i)}$ are close to the true labels $y^{(i)}$ for all training examples.

### Notation for Training Examples

For the $i$-th training example:

$$\hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$$

We can also define:

$$z^{(i)} = w^T x^{(i)} + b$$

**Notational Convention**: The superscript $(i)$ in parentheses refers to the $i$-th training example. This applies to $x^{(i)}$, $y^{(i)}$, $z^{(i)}$, etc.

## Loss Function (Single Example)

The **loss function** $\mathcal{L}(\hat{y}, y)$ measures how well the prediction $\hat{y}$ matches the true label $y$ for a **single training example**.

### Why Not Use Squared Error?

You might consider using squared error:

$$\mathcal{L}(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$$

**Problem**: This creates a **non-convex** optimization problem with multiple local optima, making gradient descent unreliable.

### Logistic Regression Loss Function

Instead, we use the **log loss** (binary cross-entropy):

$$\mathcal{L}(\hat{y}, y) = -\left[y \log(\hat{y}) + (1-y) \log(1-\hat{y})\right]$$

**Why this works**: This function is convex, making optimization much easier.

## Intuition Behind Log Loss

**Goal**: Minimize the loss function (make it as small as possible).

### Case 1: When $y = 1$

If $y = 1$, the loss becomes:

$$\mathcal{L}(\hat{y}, 1) = -\log(\hat{y})$$

(The second term vanishes because $(1-y) = 0$)

**To minimize loss**:

- Want $-\log(\hat{y})$ to be small
- This means $\log(\hat{y})$ should be large
- Therefore, $\hat{y}$ should be large
- Since $\hat{y} \in [0, 1]$, we want $\hat{y} \approx 1$

**Interpretation**: When the true label is 1, the loss pushes $\hat{y}$ close to 1.

### Case 2: When $y = 0$

If $y = 0$, the loss becomes:

$$\mathcal{L}(\hat{y}, 0) = -\log(1 - \hat{y})$$

(The first term vanishes because $y = 0$)

**To minimize loss**:

- Want $-\log(1-\hat{y})$ to be small
- This means $\log(1-\hat{y})$ should be large
- Therefore, $(1-\hat{y})$ should be large
- This means $\hat{y}$ should be small
- We want $\hat{y} \approx 0$

**Interpretation**: When the true label is 0, the loss pushes $\hat{y}$ close to 0.

### Summary

The loss function ensures:

- If $y = 1$: Push $\hat{y} \to 1$
- If $y = 0$: Push $\hat{y} \to 0$

## Cost Function (Entire Training Set)

The **loss function** measures performance on a single example. The **cost function** $J(w, b)$ measures performance on the **entire training set**.

**Definition**:

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

**Expanded form**:

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})\right]$$

Where:

- $m$ = number of training examples
- $\hat{y}^{(i)}$ = prediction for example $i$ using parameters $w$ and $b$
- $y^{(i)}$ = true label for example $i$

## Key Terminology

**Loss Function** $\mathcal{L}(\hat{y}, y)$:

- Measures error on a **single training example**
- Compares one prediction to one true label

**Cost Function** $J(w, b)$:

- Measures average error across the **entire training set**
- This is what we minimize during training

## Training Objective

**Goal**: Find parameters $w$ and $b$ that minimize the cost function $J(w, b)$.

This will be accomplished using **gradient descent**, which we'll explore in the next lesson.

## Connection to Neural Networks

Logistic regression can be viewed as a very simple neural network with:

- No hidden layers
- Single output unit
- Sigmoid activation

This makes it an excellent foundation for understanding more complex neural networks.
