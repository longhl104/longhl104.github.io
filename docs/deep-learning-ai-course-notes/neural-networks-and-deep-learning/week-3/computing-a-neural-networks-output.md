---
title: Computing a Neural Network's Output
parent: Week 3 - Shallow Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 3
last_modified_date: 2025-11-22 09:22:00 +1100
---

# Computing a Neural Network's Output
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

Now that we understand the [structure of a neural network]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/neural-network-representation.md %}), let's dive into the actual computations. The key insight: **a neural network is just logistic regression repeated multiple times** - once for each node in each layer.

## Computing One Node: Building Block

Let's start by understanding what a single node (neuron) in the hidden layer computes.

### First Hidden Unit (Node 1)

Each node performs two steps, identical to [logistic regression]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/logistic-regression.md %}):

![Neural network diagram showing computation for a single node and a two-layer network. Left side displays a single neuron receiving three inputs x1, x2, x3, computing z equals w transpose x plus b, then applying activation function sigma of z to produce output a equals y-hat. Right side shows a two-layer network with three input nodes x1, x2, x3 connecting to two hidden layer nodes, which then connect to an output node. Handwritten annotations show layer 1 computations: z1 superscript bracket 1 equals w1 transpose x plus b1, a1 equals sigma of z1; and layer 2 computations: z2 superscript bracket 2 equals w1 transpose x plus b2, a2 equals sigma of z2. Notes indicate superscript bracket ell denotes layer and subscript i denotes node in layer.](/assets/images/deep-learning/neural-networks/week-3/neural-network-diagram.png)

**Step 1: Compute linear combination**

$$z^{[1]}_1 = (w^{[1]}_1)^T x + b^{[1]}_1$$

**Step 2: Apply activation function**

$$a^{[1]}_1 = \sigma(z^{[1]}_1)$$

**Notation reminder:**

- Superscript $[1]$: layer number (hidden layer)
- Subscript $1$: node number (first node)

### Second Hidden Unit (Node 2)

The second node follows the same pattern with different parameters:

$$z^{[1]}_2 = (w^{[1]}_2)^T x + b^{[1]}_2$$

$$a^{[1]}_2 = \sigma(z^{[1]}_2)$$

### All Four Hidden Units

![Neural network diagram showing computation flow for four hidden units in the first layer. Three input features x1, x2, and x3 are on the left, connecting via arrows to four circular nodes labeled a1 superscript bracket 1, a2 superscript bracket 1, a3 superscript bracket 1, and a4 superscript bracket 1 arranged vertically in the middle. These hidden layer nodes then connect via arrows to a single output node on the right, which produces y-hat. The diagram illustrates how each hidden unit receives all three input features and contributes to the final output, demonstrating the fully connected structure of a shallow neural network.](/assets/images/deep-learning/neural-networks/week-3/neural-network-four-hidden-units.png)

For a network with 4 hidden units, we have:

$$z^{[1]}_1 = (w^{[1]}_1)^T x + b^{[1]}_1, \quad a^{[1]}_1 = \sigma(z^{[1]}_1)$$

$$z^{[1]}_2 = (w^{[1]}_2)^T x + b^{[1]}_2, \quad a^{[1]}_2 = \sigma(z^{[1]}_2)$$

$$z^{[1]}_3 = (w^{[1]}_3)^T x + b^{[1]}_3, \quad a^{[1]}_3 = \sigma(z^{[1]}_3)$$

$$z^{[1]}_4 = (w^{[1]}_4)^T x + b^{[1]}_4, \quad a^{[1]}_4 = \sigma(z^{[1]}_4)$$

> **Problem**: Computing with a for-loop over each node is inefficient. Let's vectorize!

## Vectorizing the Hidden Layer

Instead of computing each node separately, we can process all nodes simultaneously using matrix operations.

### Creating the Weight Matrix

Stack the weight vectors as **rows** in a matrix:

$$W^{[1]} = \begin{bmatrix}
— (w^{[1]}_1)^T — \\
— (w^{[1]}_2)^T — \\
— (w^{[1]}_3)^T — \\
— (w^{[1]}_4)^T —
\end{bmatrix}$$

This creates a $(4 \times 3)$ matrix where:
- 4 rows = 4 hidden units
- 3 columns = 3 input features

### Matrix Multiplication

Now compute all $z$ values at once:

$$W^{[1]} \cdot x = \begin{bmatrix}
(w^{[1]}_1)^T x \\
(w^{[1]}_2)^T x \\
(w^{[1]}_3)^T x \\
(w^{[1]}_4)^T x
\end{bmatrix}$$

### Adding Bias Vector

Stack biases vertically:

$$b^{[1]} = \begin{bmatrix}
b^{[1]}_1 \\
b^{[1]}_2 \\
b^{[1]}_3 \\
b^{[1]}_4
\end{bmatrix}$$

### Complete Hidden Layer Computation

**Linear step:**

$$z^{[1]} = W^{[1]} x + b^{[1]}$$

where $z^{[1]} = \begin{bmatrix} z^{[1]}_1 \\ z^{[1]}_2 \\ z^{[1]}_3 \\ z^{[1]}_4 \end{bmatrix}$

**Activation step:**

$$a^{[1]} = \sigma(z^{[1]})$$

The sigmoid function is applied **element-wise** to the vector $z^{[1]}$.

### Vectorization Rule of Thumb

> **Key principle**: Different nodes in a layer are **stacked vertically** in column vectors.

## Computing the Output Layer

The output layer follows the same pattern:

$$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$$

$$a^{[2]} = \sigma(z^{[2]})$$

where:
- $W^{[2]}$: shape $(1 \times 4)$ - one output unit receiving from 4 hidden units
- $b^{[2]}$: shape $(1 \times 1)$ - scalar bias
- $a^{[1]}$: shape $(4 \times 1)$ - activations from hidden layer

The output $a^{[2]}$ is our prediction: $\hat{y} = a^{[2]}$

> **Insight**: The output layer is just one logistic regression unit that takes the hidden layer activations as its input.

## Complete Forward Propagation (Single Example)

### Using Input Notation $x$

**Hidden layer:**
```python
Z1 = np.dot(W1, x) + b1    # Shape: (4, 1) = (4, 3) @ (3, 1) + (4, 1)
A1 = sigmoid(Z1)            # Shape: (4, 1)
```

**Output layer:**
```python
Z2 = np.dot(W2, A1) + b2   # Shape: (1, 1) = (1, 4) @ (4, 1) + (1, 1)
A2 = sigmoid(Z2)            # Shape: (1, 1)
```

**Prediction:**
```python
y_hat = A2
```

### Using Activation Notation $a^{[0]}$

Recall that $x = a^{[0]}$ (input layer activations). We can equivalently write:

$$z^{[1]} = W^{[1]} a^{[0]} + b^{[1]}$$

$$a^{[1]} = \sigma(z^{[1]})$$

$$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$$

$$a^{[2]} = \sigma(z^{[2]}) = \hat{y}$$

## Dimension Analysis

Let's verify the matrix dimensions work correctly:

| Computation | Matrix Dimensions | Result |
|-------------|-------------------|--------|
| $W^{[1]} x$ | $(4 \times 3) \cdot (3 \times 1)$ | $(4 \times 1)$ |
| $z^{[1]} = W^{[1]} x + b^{[1]}$ | $(4 \times 1) + (4 \times 1)$ | $(4 \times 1)$ |
| $a^{[1]} = \sigma(z^{[1]})$ | $\sigma((4 \times 1))$ | $(4 \times 1)$ |
| $W^{[2]} a^{[1]}$ | $(1 \times 4) \cdot (4 \times 1)$ | $(1 \times 1)$ |
| $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$ | $(1 \times 1) + (1 \times 1)$ | $(1 \times 1)$ |
| $a^{[2]} = \sigma(z^{[2]})$ | $\sigma((1 \times 1))$ | $(1 \times 1)$ |

All dimensions are compatible! ✓

## Comparison with Logistic Regression

| Aspect | Logistic Regression | Neural Network |
|--------|---------------------|----------------|
| Computation | $z = w^T x + b$<br>$a = \sigma(z)$ | Hidden: $z^{[1]} = W^{[1]} x + b^{[1]}, a^{[1]} = \sigma(z^{[1]})$<br>Output: $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}, a^{[2]} = \sigma(z^{[2]})$ |
| Lines of code | 2 | 4 |
| Parameters | $w, b$ | $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$ |
| Layers | 1 (output only) | 2 (hidden + output) |

## Implementation Summary

To compute the output of a 2-layer neural network for a **single example**, you need just **4 lines**:

```python
# Hidden layer
Z1 = np.dot(W1, x) + b1
A1 = sigmoid(Z1)

# Output layer
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)

# Prediction
y_hat = A2
```

## Key Takeaways

1. Each node performs two steps: **linear combination** ($z$) then **activation** ($a$)
2. A neural network is **logistic regression repeated** for each node in each layer
3. **Vectorization** eliminates for-loops by processing all nodes simultaneously
4. Stack weight vectors as **rows** in $W^{[l]}$ to enable matrix multiplication
5. Stack node outputs **vertically** in column vectors $z^{[l]}$ and $a^{[l]}$
6. Forward propagation for one example requires only **4 equations** (or 4 lines of code)
7. Matrix dimensions must be compatible: $(n^{[l]} \times n^{[l-1]}) \cdot (n^{[l-1]} \times 1) = (n^{[l]} \times 1)$
