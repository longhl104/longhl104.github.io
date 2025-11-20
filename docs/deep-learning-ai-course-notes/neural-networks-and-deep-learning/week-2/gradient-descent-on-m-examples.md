---
title: Gradient Descent on m Examples
parent: Week 2 - Neural Networks Basics
grand_parent: Neural Networks and Deep Learning
nav_order: 6
last_modified_date: 2025-11-19 11:56:00 +1100
---

# Gradient Descent
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

Previously, we learned [gradient descent for one training example]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/logistic-regression-gradient-descent.md %}). Now we'll extend this to **$m$ training examples** (the entire training set).

## Recap: Cost Function

The cost function for logistic regression:

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(a^{(i)}, y^{(i)})$$

Where:

- $m$ = number of training examples
- $a^{(i)} = \sigma(z^{(i)}) = \sigma(w^T x^{(i)} + b)$ = prediction for example $i$
- $\mathcal{L}(a^{(i)}, y^{(i)})$ = loss for example $i$

## Key Insight: Averaging Derivatives

Since the cost function is an **average** of individual losses, the derivative is also an **average**:

$$\frac{\partial J}{\partial w_1} = \frac{1}{m} \sum_{i=1}^{m} \frac{\partial \mathcal{L}^{(i)}}{\partial w_1}$$

**What this means**:

1. Compute derivative for each training example
2. Sum them up
3. Divide by $m$ to get the average

## Algorithm: Gradient Descent for $m$ Examples

### Step 1: Initialize Accumulators

```python
J = 0       # Cost accumulator
dw1 = 0     # Gradient accumulator for w1
dw2 = 0     # Gradient accumulator for w2
db = 0      # Gradient accumulator for b
```

### Step 2: Loop Over Training Examples

```python
for i in range(1, m+1):
    # Forward propagation
    z_i = w1*x1_i + w2*x2_i + b
    a_i = sigmoid(z_i)
    
    # Accumulate cost
    J += -(y_i * log(a_i) + (1-y_i) * log(1-a_i))
    
    # Backward propagation
    dz_i = a_i - y_i
    
    # Accumulate gradients
    dw1 += x1_i * dz_i
    dw2 += x2_i * dz_i
    db += dz_i
```

**Note**: This example uses 2 features. For $n$ features, you'd have `dw1`, `dw2`, ..., `dwn`.

### Step 3: Average the Gradients

```python
J = J / m       # Average cost
dw1 = dw1 / m   # Average gradient for w1
dw2 = dw2 / m   # Average gradient for w2
db = db / m     # Average gradient for b
```

### Step 4: Update Parameters

```python
w1 = w1 - alpha * dw1
w2 = w2 - alpha * dw2
b = b - alpha * db
```

Where $\alpha$ is the learning rate.

## Understanding the Variables

### Accumulators vs Per-Example Derivatives

**Accumulators** (no superscript):

- `dw1`, `dw2`, `db` - Sum across all examples
- Used to compute final gradients

**Per-example derivatives** (with superscript $i$):

- $dz^{(i)}$ - Derivative for example $i$ only
- Computed inside the loop

**Why the distinction?**

- Accumulators collect information from all examples
- Per-example derivatives are temporary calculations

## Complete One-Step Algorithm

This is **one iteration** of gradient descent:

```python
# Initialize
J, dw1, dw2, db = 0, 0, 0, 0

# Loop over training examples
for i in range(1, m+1):
    # Forward pass
    z_i = w1*x1_i + w2*x2_i + b
    a_i = sigmoid(z_i)
    J += -(y_i * log(a_i) + (1-y_i) * log(1-a_i))
    
    # Backward pass
    dz_i = a_i - y_i
    dw1 += x1_i * dz_i
    dw2 += x2_i * dz_i
    db += dz_i

# Average
J /= m
dw1 /= m
dw2 /= m
db /= m

# Update
w1 -= alpha * dw1
w2 -= alpha * dw2
b -= alpha * db
```

**To train**: Repeat this entire process many times until convergence.

## Problem: Two For-Loops

This implementation has **two weaknesses**:

### Loop 1: Over Training Examples

```python
for i in range(1, m+1):  # Loop over m examples
    ...
```

### Loop 2: Over Features

For $n$ features, you need:

```python
dw1 += x1_i * dz_i
dw2 += x2_i * dz_i
# ... 
dwn += xn_i * dz_i
```

This is essentially another loop over features.

## Why For-Loops Are a Problem

**Inefficiency in Deep Learning**:

**Pre-Deep Learning Era**:

- For-loops were acceptable
- Vectorization was a "nice to have"
- Small datasets made speed less critical

**Deep Learning Era**:

- Massive datasets (millions of examples)
- For-loops are too slow
- Vectorization is **essential**

**The Solution**: **Vectorization** - techniques that eliminate explicit for-loops

## Performance Comparison

| Approach | Speed | Scalability |
|----------|-------|-------------|
| **Two for-loops** | Slow | Poor (doesn't scale) |
| **Vectorized** | Fast | Excellent (scales to millions) |

**Why it matters**:

- Modern datasets: millions of examples
- Need to process quickly
- For-loops don't scale

## What's Next: Vectorization

In the next videos, we'll learn **vectorization techniques** to:

1. **Eliminate the loop over training examples**
2. **Eliminate the loop over features**
3. **Implement gradient descent with no explicit loops**

**Benefits of vectorization**:

- Much faster execution
- Cleaner code
- Scales to massive datasets
- Takes advantage of modern hardware (GPUs)

## Key Takeaways

1. **Gradient for multiple examples**: Average of individual gradients
2. **Accumulators**: Sum gradients across examples, then divide by $m$
3. **One iteration**: Forward pass → compute gradients → average → update
4. **Multiple iterations**: Repeat until convergence
5. **For-loops are slow**: Need vectorization for efficiency
6. **Next step**: Learn vectorization to eliminate loops

## Summary Formula

For the entire training set:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)} (a^{(i)} - y^{(i)})$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (a^{(i)} - y^{(i)})$$

These formulas compute the gradients needed for gradient descent across all examples.
