---
title: Vectorizing Logistic Regression
parent: Week 2 - Neural Networks Basics
grand_parent: Neural Networks and Deep Learning
nav_order: 9
last_modified_date: 2025-11-20 21:20:00 +1100
---

# Vectorizing Logistic Regression
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

We've seen how [vectorization speeds up code]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/vectorization.md %}) and how to [eliminate one for-loop]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/more-vectorization-examples.md %}). Now we'll **eliminate the remaining for-loop** to process an entire training set with **zero explicit loops**.

**Goal**: Implement one complete iteration of gradient descent for logistic regression without any for-loops.

**Why this matters**: This technique is fundamental to efficient neural network implementation.

## The Problem: Forward Propagation with Loops

### Current Approach (One Loop)

For $m$ training examples, we compute predictions sequentially:

**Example 1**:
$$z^{(1)} = w^T x^{(1)} + b$$
$$a^{(1)} = \sigma(z^{(1)})$$

**Example 2**:
$$z^{(2)} = w^T x^{(2)} + b$$
$$a^{(2)} = \sigma(z^{(2)})$$

**Example 3**:
$$z^{(3)} = w^T x^{(3)} + b$$
$$a^{(3)} = \sigma(z^{(3)})$$

⋮

**Example m**:
$$z^{(m)} = w^T x^{(m)} + b$$
$$a^{(m)} = \sigma(z^{(m)})$$

**Current code**:

```python
for i in range(m):
    z_i = np.dot(w.T, x[:,i]) + b
    a_i = sigmoid(z_i)
```

**Challenge**: Can we compute all $z^{(i)}$ and $a^{(i)}$ simultaneously?

## The Solution: Matrix Representation

### Recall: Training Data Matrix $X$

From our [notation section]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-1/what-is-neural-network.md %}), we defined:

$$X = \begin{bmatrix}
| & | & & | \\
x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\
| & | & & |
\end{bmatrix}$$

**Dimensions**: $X \in \mathbb{R}^{n_x \times m}$
- Each column is one training example
- $n_x$ features (rows)
- $m$ examples (columns)

**NumPy shape**: `X.shape = (n_x, m)`

## Vectorizing Forward Propagation

### Step 1: Compute All $z^{(i)}$ at Once

**Individual computations**:
$$z^{(1)} = w^T x^{(1)} + b$$
$$z^{(2)} = w^T x^{(2)} + b$$
$$\vdots$$
$$z^{(m)} = w^T x^{(m)} + b$$

**Vectorized form**:
$$Z = w^T X + b$$

Where:
$$Z = \begin{bmatrix} z^{(1)} & z^{(2)} & \cdots & z^{(m)} \end{bmatrix}$$

**Dimensions**: $Z \in \mathbb{R}^{1 \times m}$ (row vector)

### Understanding the Matrix Multiplication

**Expanding $w^T X$**:

$$w^T X = w^T \begin{bmatrix} x^{(1)} & x^{(2)} & \cdots & x^{(m)} \end{bmatrix}$$

$$= \begin{bmatrix} w^T x^{(1)} & w^T x^{(2)} & \cdots & w^T x^{(m)} \end{bmatrix}$$

**Adding bias $b$**:

$$w^T X + b = \begin{bmatrix} w^T x^{(1)} + b & w^T x^{(2)} + b & \cdots & w^T x^{(m)} + b \end{bmatrix}$$

$$= \begin{bmatrix} z^{(1)} & z^{(2)} & \cdots & z^{(m)} \end{bmatrix} = Z$$

### Python Implementation

```python
Z = np.dot(w.T, X) + b
```

**Breaking it down**:
- `w.T` has shape `(1, n_x)` - row vector
- `X` has shape `(n_x, m)` - matrix
- `np.dot(w.T, X)` has shape `(1, m)` - row vector
- `b` is a scalar (or shape `(1, 1)`)
- Result `Z` has shape `(1, m)` - row vector

### Broadcasting in Python

**Important subtlety**: When we write `Z = np.dot(w.T, X) + b`, where:
- `np.dot(w.T, X)` is a `(1, m)` matrix
- `b` is a scalar

**What happens**: Python automatically **broadcasts** (expands) `b`:

$$b \rightarrow \begin{bmatrix} b & b & \cdots & b \end{bmatrix}$$

This creates a `(1, m)` row vector with $b$ repeated $m$ times.

**Result**: Element-wise addition works correctly:

$$\begin{bmatrix} w^T x^{(1)} & w^T x^{(2)} & \cdots & w^T x^{(m)} \end{bmatrix} + \begin{bmatrix} b & b & \cdots & b \end{bmatrix}$$

$$= \begin{bmatrix} w^T x^{(1)} + b & w^T x^{(2)} + b & \cdots & w^T x^{(m)} + b \end{bmatrix}$$

**Note**: Broadcasting is covered in detail in the next video.

## Step 2: Compute All $a^{(i)}$ at Once

### Individual Activations

$$a^{(1)} = \sigma(z^{(1)})$$
$$a^{(2)} = \sigma(z^{(2)})$$
$$\vdots$$
$$a^{(m)} = \sigma(z^{(m)})$$

### Vectorized Form

Define:
$$A = \begin{bmatrix} a^{(1)} & a^{(2)} & \cdots & a^{(m)} \end{bmatrix}$$

**Compute**:
$$A = \sigma(Z)$$

Where $\sigma$ is applied **element-wise** to the entire matrix.

### Python Implementation

```python
A = sigmoid(Z)
```

**Requirements**: The `sigmoid` function must handle vectors/matrices:

```python
def sigmoid(Z):
    """
    Applies sigmoid element-wise to Z.
    Works for scalars, vectors, or matrices.
    """
    return 1 / (1 + np.exp(-Z))
```

**Result**: `A` has shape `(1, m)` containing all predictions.

## Complete Vectorized Forward Propagation

### From This (Loop-based):

```python
for i in range(m):
    z_i = np.dot(w.T, x[:,i]) + b
    a_i = sigmoid(z_i)
    # Store z_i and a_i somewhere...
```

### To This (Vectorized):

```python
Z = np.dot(w.T, X) + b  # All z values at once
A = sigmoid(Z)           # All activations at once
```

**Comparison**:

| Approach | Lines of Code | Loops | Speed |
|----------|---------------|-------|-------|
| **Loop-based** | ~5 lines | 1 for-loop | Slow |
| **Vectorized** | 2 lines | 0 for-loops | Fast |

## Visualizing the Stacking Pattern

**Pattern**: Just as we stack examples, we stack results.

### Input Stacking
$$X = \begin{bmatrix} x^{(1)} & x^{(2)} & \cdots & x^{(m)} \end{bmatrix}$$

### Output Stacking
$$Z = \begin{bmatrix} z^{(1)} & z^{(2)} & \cdots & z^{(m)} \end{bmatrix}$$

$$A = \begin{bmatrix} a^{(1)} & a^{(2)} & \cdots & a^{(m)} \end{bmatrix}$$

**Key insight**: Each operation preserves the column structure—column $i$ of output corresponds to example $i$.

## Summary: Vectorized Forward Pass

**Complete implementation**:

```python
# Input: X (n_x, m), w (n_x, 1), b (scalar)
Z = np.dot(w.T, X) + b  # Shape: (1, m)
A = sigmoid(Z)           # Shape: (1, m)
```

**Benefits**:
- ✅ No explicit for-loops
- ✅ Processes all $m$ examples simultaneously
- ✅ Much faster execution
- ✅ Cleaner code

**What we've computed**:
- `Z[0, i]` = $z^{(i)}$ for example $i$
- `A[0, i]` = $a^{(i)}$ = $\hat{y}^{(i)}$ for example $i$

## What's Next: Backward Propagation

We've vectorized **forward propagation**. But gradient descent also needs **backward propagation** (computing gradients).

**Next video**: Learn how to vectorize backward propagation to compute:
- $\frac{\partial J}{\partial w}$
- $\frac{\partial J}{\partial b}$

All without loops!

## Key Takeaways

1. **Matrix multiplication** computes all examples simultaneously
2. **Stacking pattern**: Columns of $X$ → columns of $Z$ → columns of $A$
3. **Broadcasting** automatically expands scalars to match dimensions
4. **Two lines of code** replace an entire for-loop
5. **Element-wise functions** (`sigmoid`, `exp`) work on entire matrices
6. **Next step**: Vectorize backward propagation for complete loop-free implementation

**Remember**: Vectorization is essential for efficient deep learning—this technique scales to millions of examples.
