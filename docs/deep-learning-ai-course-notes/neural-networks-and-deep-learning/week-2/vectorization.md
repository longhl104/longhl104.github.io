---
title: Vectorization
parent: Week 2 - Neural Networks Basics
grand_parent: Neural Networks and Deep Learning
nav_order: 7
last_modified_date: 2025-11-20 13:53:00 +1100
---

# Vectorization
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

**Vectorization** is the art of eliminating explicit for-loops in your code. In deep learning, this is a **critical skill** because:

- Deep learning works best with **large datasets**
- For-loops make training **extremely slow**
- Vectorized code can run **300x faster**

**Why it matters**: The difference between your code taking 1 minute vs 5 hours.

## What is Vectorization?

### The Problem: Computing $z = w^T x + b$

In logistic regression, we need to compute:

$$z = w^T x + b$$

Where:

- $w \in \mathbb{R}^{n_x}$ (weight vector)
- $x \in \mathbb{R}^{n_x}$ (feature vector)
- Both can be very large vectors (many features)

### Non-Vectorized Implementation (Slow)

```python
z = 0
for i in range(n_x):
    z += w[i] * x[i]
z += b
```

**Problem**: This explicit for-loop is **very slow** for large $n_x$.

### Vectorized Implementation (Fast)

```python
z = np.dot(w, x) + b
```

**Benefits**:

- Single line of code
- No explicit loop
- **Much faster execution**

## Performance Comparison Demo

Let's demonstrate the speed difference with a concrete example.

### Setup: Create Test Data

```python
import numpy as np
import time

# Create two 1-million dimensional arrays
a = np.random.rand(1000000)
b = np.random.rand(1000000)
```

### Vectorized Version

```python
tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(f"Vectorized version: {1000*(toc-tic):.2f} ms")
```

**Result**: ~1.5 milliseconds

### Non-Vectorized Version

```python
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()

print(f"For loop version: {1000*(toc-tic):.2f} ms")
print(f"Result: {c}")
```

**Result**: ~481 milliseconds

### Performance Summary

| Implementation | Time | Speedup |
|---------------|------|---------|
| **Vectorized** | 1.5 ms | 1x (baseline) |
| **For-loop** | 481 ms | **321x slower** |

**Both compute the same result**, but vectorized is ~300x faster!

## Why Vectorization Works: SIMD

### Hardware Parallelization

Both CPUs and GPUs have **SIMD instructions**:

**SIMD** = **Single Instruction Multiple Data**

**What this means**:

- Process multiple data points simultaneously
- One operation → many calculations in parallel

### Where Vectorization Runs

| Hardware | Performance | Notes |
|----------|-------------|-------|
| **GPU** | Excellent | Specialized for parallel computation |
| **CPU** | Very Good | Also supports SIMD, just not as optimized |

**Key insight**: NumPy's built-in functions (`np.dot`, `np.sum`, etc.) automatically leverage SIMD parallelism on both CPUs and GPUs.

## Why This Matters in Deep Learning

### Training Time Comparison

**Non-vectorized code**:

- Small dataset: manageable
- Large dataset: hours or days

**Vectorized code**:

- Small dataset: instant
- Large dataset: minutes

### Real-World Impact

| Code Type | 1M Examples | 10M Examples |
|-----------|-------------|--------------|
| **For-loops** | 5 hours | 50 hours |
| **Vectorized** | 1 minute | 10 minutes |

## The Golden Rule

> **Whenever possible, avoid explicit for-loops**

**Instead**:

- Use NumPy's built-in functions
- Think in terms of matrix/vector operations
- Let the library handle parallelization

## Common Vectorization Patterns

### Instead of This (Loop)

```python
result = 0
for i in range(n):
    result += w[i] * x[i]
```

### Do This (Vectorized)

```python
result = np.dot(w, x)
```

### More Examples

| Operation | Loop Version | Vectorized |
|-----------|-------------|------------|
| Dot product | `sum(w[i]*x[i])` | `np.dot(w, x)` |
| Element-wise multiply | `[w[i]*x[i] for i]` | `w * x` |
| Sum | `sum(x[i] for i)` | `np.sum(x)` |
| Exponential | `[exp(x[i]) for i]` | `np.exp(x)` |

## Key Takeaways

1. **Vectorization eliminates for-loops** using matrix operations
2. **300x speedup** is typical for vectorized vs loop-based code
3. **SIMD instructions** enable parallel processing on CPUs and GPUs
4. **NumPy functions** automatically leverage hardware parallelization
5. **Deep learning requires vectorization** to handle large datasets efficiently
6. **Rule of thumb**: Always prefer vectorized operations over loops

**Remember**: In deep learning, vectorization isn't optional—it's essential for practical implementation.
