---
title: Vectorizing Logistic Regression's Gradient Output
parent: Week 2 - Neural Networks Basics
grand_parent: Neural Networks and Deep Learning
nav_order: 10
last_modified_date: 2025-11-20 21:32:00 +1100
---

# Vectorizing Logistic Regression's Gradient Output
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

In the [previous post]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/vectorizing-logistic-regression.md %}#vectorizing-forward-propagation), you learned how to vectorize **forward propagation** to compute predictions for all training examples simultaneously. Now we'll vectorize **backward propagation** to compute gradients for all $m$ examples at once.

**Goal**: Complete the fully vectorized implementation of logistic regression with zero explicit loops.

## Recap: Individual Gradient Computations

For gradient computation, we previously computed for each example:

$$dz^{(1)} = a^{(1)} - y^{(1)}$$
$$dz^{(2)} = a^{(2)} - y^{(2)}$$
$$\vdots$$
$$dz^{(m)} = a^{(m)} - y^{(m)}$$

**Problem**: This requires looping over $m$ examples.

## Vectorizing $dZ$: Stacking the Derivatives

### Define Matrix $dZ$

Just as we stacked $z$ values and $a$ values horizontally, we stack $dz$ values:

$$dZ = \begin{bmatrix} dz^{(1)} & dz^{(2)} & \cdots & dz^{(m)} \end{bmatrix}$$

**Dimensions**: $dZ \in \mathbb{R}^{1 \times m}$ (row vector)

### Recall from Forward Propagation

We already computed:

$$A = \begin{bmatrix} a^{(1)} & a^{(2)} & \cdots & a^{(m)} \end{bmatrix}$$

$$Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix}$$

### One-Line Computation

**Key insight**: 

$$dZ = A - Y$$

**Why this works**:

$$A - Y = \begin{bmatrix} a^{(1)} - y^{(1)} & a^{(2)} - y^{(2)} & \cdots & a^{(m)} - y^{(m)} \end{bmatrix}$$

$$= \begin{bmatrix} dz^{(1)} & dz^{(2)} & \cdots & dz^{(m)} \end{bmatrix} = dZ$$

**Result**: With one line of code, we compute all $dz$ values simultaneously!

## The Remaining Loop Problem

### Where We Were (One Loop Remaining)

After [eliminating the features loop]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/more-vectorization-examples.md %}), we still had:

```python
# Initialize
dw = np.zeros((n_x, 1))
db = 0

# Loop over m training examples (still needed!)
for i in range(m):
    dw += x[:,i] * dz[i]
    db += dz[i]

# Average
dw /= m
db /= m
```

**Problem**: We're looping over all $m$ examples.

**Challenge**: Can we eliminate this last loop?

## Vectorizing $db$: Gradient for Bias

### What We're Computing

The bias gradient is:

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} dz^{(i)}$$

### Non-Vectorized (Loop)

```python
db = 0
for i in range(m):
    db += dz[i]
db /= m
```

### Vectorized (No Loop)

```python
db = (1/m) * np.sum(dZ)
```

**Explanation**: 
- `dZ` contains all $dz^{(i)}$ values in a row vector
- `np.sum(dZ)` adds them all up
- Divide by $m$ to get the average

## Vectorizing $dw$: Gradient for Weights

### What We're Computing

The weight gradient is:

$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} x^{(i)} dz^{(i)}$$

### Matrix Formulation

**Formula**:

$$dw = \frac{1}{m} X \cdot dZ^T$$

**Why this works**: Let's expand the matrix multiplication.

**Matrix dimensions**:
- $X$ has shape $(n_x, m)$ - training examples as columns
- $dZ^T$ has shape $(m, 1)$ - transpose of row vector

**Expanding the multiplication**:

$$X \cdot dZ^T = \begin{bmatrix} 
| & | & & | \\
x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\
| & | & & |
\end{bmatrix} \cdot \begin{bmatrix} dz^{(1)} \\ dz^{(2)} \\ \vdots \\ dz^{(m)} \end{bmatrix}$$

$$= x^{(1)} dz^{(1)} + x^{(2)} dz^{(2)} + \cdots + x^{(m)} dz^{(m)}$$

**Result**: A column vector of shape $(n_x, 1)$ - exactly what we need for $dw$!

### Vectorized Implementation

```python
dw = (1/m) * np.dot(X, dZ.T)
```

**One line of code** replaces the entire loop over training examples!

## Summary: Vectorized Gradient Computation

**Complete backward propagation**:

```python
dZ = A - Y                      # Shape: (1, m)
dw = (1/m) * np.dot(X, dZ.T)   # Shape: (n_x, 1)
db = (1/m) * np.sum(dZ)        # Scalar
```

**No loops needed!**

## Complete Vectorized Logistic Regression

### Original (Highly Inefficient - Two Loops)

```python
# Initialize
J = 0
dw = np.zeros((n_x, 1))
db = 0

# Loop 1: Over training examples
for i in range(m):
    # Forward propagation
    z_i = np.dot(w.T, x[:,i]) + b
    a_i = sigmoid(z_i)
    
    # Cost accumulation
    J += -(y[i] * np.log(a_i) + (1-y[i]) * np.log(1-a_i))
    
    # Loop 2: Over features (implicit in dw1, dw2, etc.)
    dz_i = a_i - y[i]
    dw += x[:,i] * dz_i
    db += dz_i

# Average
J /= m
dw /= m
db /= m

# Update
w -= alpha * dw
b -= alpha * db
```

### After Step 1: Eliminated Features Loop

(From the [previous post]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/more-vectorization-examples.md %}#eliminating-loop-2-vectorize-over-features))

```python
# Initialize
dw = np.zeros((n_x, 1))  # Vector, not separate dw1, dw2, etc.
db = 0

# Loop over training examples (still here!)
for i in range(m):
    z_i = np.dot(w.T, x[:,i]) + b
    a_i = sigmoid(z_i)
    dz_i = a_i - y[i]
    
    dw += x[:,i] * dz_i  # Vector operation
    db += dz_i

# Average and update
dw /= m
db /= m
w -= alpha * dw
b -= alpha * db
```

### Final: Fully Vectorized (Zero Loops!)

```python
# Forward propagation
Z = np.dot(w.T, X) + b    # Shape: (1, m)
A = sigmoid(Z)             # Shape: (1, m)

# Backward propagation
dZ = A - Y                 # Shape: (1, m)
dw = (1/m) * np.dot(X, dZ.T)  # Shape: (n_x, 1)
db = (1/m) * np.sum(dZ)    # Scalar

# Update parameters
w -= alpha * dw
b -= alpha * db
```

**Result**: One iteration of gradient descent with **zero explicit loops**!

## Implementation Steps Summary

| Step | Code | What It Does |
|------|------|--------------|
| **Forward** | `Z = np.dot(w.T, X) + b` | Compute all $z^{(i)}$ |
| | `A = sigmoid(Z)` | Compute all $a^{(i)}$ |
| **Backward** | `dZ = A - Y` | Compute all $dz^{(i)}$ |
| | `dw = (1/m) * np.dot(X, dZ.T)` | Compute $\frac{\partial J}{\partial w}$ |
| | `db = (1/m) * np.sum(dZ)` | Compute $\frac{\partial J}{\partial b}$ |
| **Update** | `w -= alpha * dw` | Update weights |
| | `b -= alpha * db` | Update bias |

## One Exception: Iterations Loop

**Important note**: To train the model, you still need a loop over **gradient descent iterations**.

```python
for iteration in range(num_iterations):
    # Forward propagation
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    
    # Backward propagation
    dZ = A - Y
    dw = (1/m) * np.dot(X, dZ.T)
    db = (1/m) * np.sum(dZ)
    
    # Update
    w -= alpha * dw
    b -= alpha * db
```

**Why we can't eliminate this loop**: 
- Each iteration depends on the previous update
- Must be done sequentially
- This is inherent to gradient descent

**What we achieved**: 
- ✅ No loop over training examples
- ✅ No loop over features
- ❌ Still need loop over iterations (unavoidable)

## Comparison: Loop vs Vectorized

### Speed Comparison

| Approach | Training Examples Loop | Features Loop | Relative Speed |
|----------|----------------------|---------------|----------------|
| **Original** | ✗ (explicit) | ✗ (implicit) | 1x (baseline) |
| **Partial** | ✗ (explicit) | ✓ (vectorized) | ~10x faster |
| **Full** | ✓ (vectorized) | ✓ (vectorized) | **~300x faster** |

### Code Comparison

| Metric | Loop-Based | Vectorized |
|--------|-----------|------------|
| **Lines of code** | ~15 lines | 7 lines |
| **Explicit loops** | 1-2 loops | 0 loops |
| **Scalability** | Poor | Excellent |
| **Readability** | Complex | Clean |

## Key Takeaways

1. **Stacking pattern**: Stack individual values horizontally into matrices
2. **$dZ = A - Y$**: One line computes all prediction errors
3. **$dw = \frac{1}{m} X dZ^T$**: Matrix multiplication replaces the examples loop
4. **$db = \frac{1}{m} \sum dZ$**: NumPy sum replaces accumulation loop
5. **7 lines of code**: Complete one iteration of gradient descent
6. **Iteration loop remains**: Inherent to gradient descent, can't be eliminated
7. **~300x speedup**: Fully vectorized vs loop-based implementation
8. **Essential for deep learning**: This technique scales to millions of examples

**Remember**: Vectorization transforms logistic regression from impractical (for large datasets) to highly efficient!
