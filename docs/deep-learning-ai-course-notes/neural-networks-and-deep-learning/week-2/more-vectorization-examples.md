---
title: More Vectorization Examples
parent: Week 2 - Neural Networks Basics
grand_parent: Neural Networks and Deep Learning
nav_order: 8
last_modified_date: 2025-11-20 14:05:00 +1100
---

# More Vectorization Examples
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

In the [previous post]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/vectorization.md %}), you saw how vectorization speeds up code by using built-in functions and avoiding explicit for-loops. Let's explore more examples and apply these concepts to **logistic regression**.

## The Golden Rule (Revisited)

> **Whenever possible, avoid explicit for-loops**

**Why?**

- Built-in functions are much faster
- Code is cleaner and more readable
- Better hardware utilization

**Caveat**: It's not always possible to eliminate every loop, but you should always look for vectorization opportunities first.

## Example 1: Matrix-Vector Multiplication

### Problem: Compute $u = Av$

**Mathematical definition**:

$$u_i = \sum_{j} A_{ij} v_j$$

### Non-Vectorized Implementation (Slow)

```python
# Initialize result vector
u = np.zeros((n, 1))

# Nested loops over i and j
for i in range(n):
    for j in range(n):
        u[i] += A[i][j] * v[j]
```

**Problems**:

- Two nested for-loops
- Very slow for large matrices

### Vectorized Implementation (Fast)

```python
u = np.dot(A, v)
```

**Benefits**:

- One line of code
- Eliminates **two for-loops**
- Much faster execution

## Example 2: Element-wise Exponential

### Problem: Apply $e^x$ to Every Element

Given vector:

$$v = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

Compute:

$$u = \begin{bmatrix} e^{v_1} \\ e^{v_2} \\ \vdots \\ e^{v_n} \end{bmatrix}$$

### Non-Vectorized Implementation (Slow)

```python
# Initialize result vector
u = np.zeros((n, 1))

# Loop over each element
for i in range(n):
    u[i] = np.exp(v[i])
```

### Vectorized Implementation (Fast)

```python
import numpy as np
u = np.exp(v)
```

**Result**: One line replaces the entire for-loop!

## NumPy Vectorized Functions

NumPy provides many **element-wise operations** that eliminate for-loops:

| Operation | NumPy Function | Description |
|-----------|---------------|-------------|
| Exponential | `np.exp(v)` | $e^{v_i}$ for each element |
| Logarithm | `np.log(v)` | $\log(v_i)$ for each element |
| Absolute value | `np.abs(v)` | $\|v_i\|$ for each element |
| Maximum | `np.maximum(v, 0)` | $\max(v_i, 0)$ for each element |
| Square | `v**2` | $v_i^2$ for each element |
| Inverse | `1/v` | $1/v_i$ for each element |
| Sigmoid | `1/(1 + np.exp(-v))` | $\sigma(v_i)$ for each element |

**Key insight**: Whenever you're tempted to write a for-loop, check if there's a NumPy built-in function!

## Applying Vectorization to Logistic Regression

Recall our [gradient descent implementation]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/gradient-descent-on-m-examples.md %}) had **two for-loops**:

### Original Implementation (Two Loops)

```python
# Initialize gradients for each feature
dw1 = 0
dw2 = 0
db = 0

# Loop 1: Over training examples
for i in range(m):
    z_i = w1*x1[i] + w2*x2[i] + b
    a_i = sigmoid(z_i)
    dz_i = a_i - y[i]
    
    # Loop 2: Over features (implicit)
    dw1 += x1[i] * dz_i
    dw2 += x2[i] * dz_i
    db += dz_i

# Average
dw1 /= m
dw2 /= m
db /= m
```

**Problem**: For $n_x$ features, we need separate variables `dw1`, `dw2`, ..., `dwn`, essentially creating a hidden loop:

```python
for j in range(n_x):
    dw[j] += x[i][j] * dz_i
```

## Eliminating Loop 2: Vectorize Over Features

### Strategy: Use Vector Operations

Instead of separate `dw1`, `dw2`, etc., use a **vector** `dw`.

### Improved Implementation (One Loop)

```python
# Initialize gradient vector (not scalars!)
dw = np.zeros((n_x, 1))  # Vector for all features
db = 0

# Loop over training examples (still needed for now)
for i in range(m):
    z_i = np.dot(w.T, x[:,i]) + b  # Vectorized computation
    a_i = sigmoid(z_i)
    dz_i = a_i - y[i]
    
    # Vector operation (replaces inner loop!)
    dw += x[:,i] * dz_i
    db += dz_i

# Average (vector division)
dw /= m
db /= m
```

**Key changes**:

| Before | After |
|--------|-------|
| `dw1 = 0, dw2 = 0, ...` | `dw = np.zeros((n_x, 1))` |
| `dw1 += x1[i] * dz_i` | `dw += x[:,i] * dz_i` |
| `dw1 /= m, dw2 /= m, ...` | `dw /= m` |

### Progress: Two Loops → One Loop

**Before**: Loop over examples + loop over features
**After**: Loop over examples only

**Speedup**: Already faster! But we can do even better...

## What's Next: Eliminating the Last Loop

We still have one for-loop over training examples:

```python
for i in range(m):
    # Process example i
    ...
```

**Surprising result**: We can eliminate this loop too!

**Next video**: Learn how to process **all training examples simultaneously** without any explicit loops.

**Preview**: Instead of processing examples one at a time, we'll use matrix operations to process the entire dataset in one shot.

## Vectorization Progress Summary

| Stage | Loops | Example Code |
|-------|-------|-------------|
| **Original** | 2 loops | `for i: for j: dw[j] += ...` |
| **After this video** | 1 loop | `for i: dw += x[:,i] * dz_i` |
| **Next video** | 0 loops | `dw = (1/m) * X @ dz` |

## Key Takeaways

1. **Matrix multiplication**: Use `np.dot(A, v)` instead of nested loops
2. **Element-wise operations**: NumPy has functions like `np.exp()`, `np.log()`, etc.
3. **Vector operations**: Replace scalar variables with vectors
4. **Gradual improvement**: Eliminate loops one at a time
5. **Check for built-ins**: Before writing a loop, search for a NumPy function
6. **Next step**: Eliminate the training examples loop using full matrix operations

## Practice Checklist

When writing code, ask yourself:

- [ ] Am I using a for-loop?
- [ ] Is there a NumPy built-in function for this?
- [ ] Can I use vector/matrix operations instead?
- [ ] Have I eliminated all unnecessary loops?

**Remember**: Each loop you eliminate makes your code significantly faster!
