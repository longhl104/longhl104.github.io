---
title: Forward and Backward Propagation
parent: Week 4 - Deep Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 6
last_modified_date: 2025-11-24 14:12:00 +1100
---

# Forward and Backward Propagation
{: .no_toc }
## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

You've learned about the [building blocks of deep neural networks]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-4/building-blocks-of-deep-neural-networks.md %}) - forward and backward propagation for each layer. Now let's see the **complete implementation details** with all the equations you need.

This lesson provides:
- Complete forward propagation equations (single example + vectorized)
- Complete backward propagation equations (single example + vectorized)
- How to initialize the forward and backward passes
- A concrete 3-layer network example

## Forward Propagation Implementation

### Single Layer Forward Function

Recall from the [previous lesson]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-4/building-blocks-of-deep-neural-networks.md %}), the forward function:

**Input**: $a^{[l-1]}$ (activations from previous layer)

**Output**: $a^{[l]}$ (activations for this layer), cache

**Cache**: $z^{[l]}, W^{[l]}, b^{[l]}, a^{[l-1]}$ (needed for backprop)

### Forward Propagation Equations

#### Single Example

For a single training example:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$

$$a^{[l]} = g^{[l]}(z^{[l]})$$

Where $g^{[l]}$ is the activation function for layer $l$.

#### Vectorized Implementation

For all $m$ training examples:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

$$A^{[l]} = g^{[l]}(Z^{[l]})$$

**Note**: $b^{[l]}$ uses Python broadcasting to add to each column of $W^{[l]} A^{[l-1]}$.

### Forward Propagation Algorithm

```python
def forward_propagation_layer_l(A_prev, W, b, activation):
    """
    Implement forward propagation for layer l
    
    Args:
        A_prev: activations from previous layer, shape (n[l-1], m)
        W: weight matrix, shape (n[l], n[l-1])
        b: bias vector, shape (n[l], 1)
        activation: "relu" or "sigmoid"
    
    Returns:
        A: activations for this layer, shape (n[l], m)
        cache: tuple containing (Z, A_prev, W, b)
    """
    # Linear step
    Z = np.dot(W, A_prev) + b
    
    # Activation step
    if activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
    
    # Cache for backpropagation
    cache = (Z, A_prev, W, b)
    
    return A, cache
```

### Initializing Forward Propagation

**Starting point**: $a^{[0]} = X$

- For a single example: $a^{[0]} = x$ (input features)
- For vectorized: $A^{[0]} = X$ (entire training set)

**Chain of computations**:

```
X = A^[0] → [Layer 1] → A^[1] → [Layer 2] → A^[2] → ... → A^[L] = Ŷ
```

This left-to-right chain computes predictions by repeatedly applying the forward function.

## Backward Propagation Implementation

### Single Layer Backward Function

**Input**: $dA^{[l]}$ (gradient of cost w.r.t. activations)

**Output**: $dA^{[l-1]}$ (gradient to pass back), $dW^{[l]}$, $db^{[l]}$ (gradients for this layer's parameters)

### Backward Propagation Equations

#### The Four Key Equations (Single Example)

These four equations implement backpropagation for layer $l$:

**Equation 1: Gradient through activation function**

$$dZ^{[l]} = dA^{[l]} \odot g'^{[l]}(Z^{[l]})$$

**Equation 2: Gradient w.r.t. weights**

$$dW^{[l]} = dZ^{[l]} \cdot (a^{[l-1]})^T$$

**Equation 3: Gradient w.r.t. bias**

$$db^{[l]} = dZ^{[l]}$$

**Equation 4: Gradient w.r.t. previous activations**

$$dA^{[l-1]} = (W^{[l]})^T \cdot dZ^{[l]}$$

> **Note**: $\odot$ denotes element-wise multiplication

#### Connection to Previous Formula

If you substitute Equation 4 into Equation 1 for layer $l-1$, you get:

$$dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \odot g'^{[l]}(Z^{[l]})$$

This matches the [backpropagation equations]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/gradient-descend-for-neural-networks.md %}#backpropagation-equations) from Week 3!

#### Vectorized Implementation

For all $m$ training examples:

**Equation 1: Activation gradient**

$$dZ^{[l]} = dA^{[l]} \odot g'^{[l]}(Z^{[l]})$$

**Equation 2: Weight gradient**

$$dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$$

**Equation 3: Bias gradient**

$$db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l]}$$

(sum over training examples, keeping dimensions)

**Equation 4: Previous activation gradient**

$$dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}$$

### Backward Propagation Algorithm

```python
def backward_propagation_layer_l(dA, cache, activation):
    """
    Implement backward propagation for layer l
    
    Args:
        dA: gradient of cost w.r.t. activations, shape (n[l], m)
        cache: tuple (Z, A_prev, W, b) from forward prop
        activation: "relu" or "sigmoid"
    
    Returns:
        dA_prev: gradient w.r.t. previous activations, shape (n[l-1], m)
        dW: gradient w.r.t. weights, shape (n[l], n[l-1])
        db: gradient w.r.t. bias, shape (n[l], 1)
    """
    # Unpack cache
    Z, A_prev, W, b = cache
    m = A_prev.shape[1]
    
    # Equation 1: Activation gradient
    if activation == "relu":
        dZ = dA * relu_derivative(Z)
    elif activation == "sigmoid":
        dZ = dA * sigmoid_derivative(Z)
    
    # Equation 2: Weight gradient
    dW = (1/m) * np.dot(dZ, A_prev.T)
    
    # Equation 3: Bias gradient
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    # Equation 4: Previous activation gradient
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db
```

**NumPy implementation detail**:

```python
db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
```

- `axis=1`: Sum across training examples (columns)
- `keepdims=True`: Keep result as $(n^{[l]}, 1)$ instead of $(n^{[l]},)$

### Initializing Backward Propagation

**Starting point**: $dA^{[L]}$ (gradient of loss w.r.t. final predictions)

For **binary classification** with logistic loss:

$$\frac{\partial \mathcal{L}}{\partial a^{[L]}} = -\frac{y}{a^{[L]}} + \frac{1-y}{1-a^{[L]}}$$

#### Single Example

$$dA^{[L]} = -\frac{y}{a^{[L]}} + \frac{1-y}{1-a^{[L]}}$$

#### Vectorized (All Examples)

$$dA^{[L]} = \begin{bmatrix} -\frac{y^{(1)}}{a^{[L](1)}} + \frac{1-y^{(1)}}{1-a^{[L](1)}} & \cdots & -\frac{y^{(m)}}{a^{[L](m)}} + \frac{1-y^{(m)}}{1-a^{[L](m)}} \end{bmatrix}$$

Or more concisely in Python:

```python
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
```

**Chain of computations**:

```
dA^[L] ← [Layer L] ← dA^[L-1] ← [Layer L-1] ← ... ← dA^[1] ← [Layer 1] ← dA^[0]
   ↓                    ↓                              ↓                  ↓
dW^[L], db^[L]      dW^[L-1], db^[L-1]           dW^[1], db^[1]    (not used)
```

This right-to-left chain computes all gradients by repeatedly applying the backward function.

## Complete Example: 3-Layer Neural Network

Let's see how forward and backward propagation work together in a complete network.

### Network Architecture

```
Input X (features)
   ↓
Layer 1 (ReLU)
   ↓
Layer 2 (ReLU)
   ↓
Layer 3 (Sigmoid) → binary classification
   ↓
Output Ŷ (predictions)
```

### Forward Propagation (Left to Right)

**Layer 1**:

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

$$A^{[1]} = \text{ReLU}(Z^{[1]})$$

Cache: $(Z^{[1]}, X, W^{[1]}, b^{[1]})$

**Layer 2**:

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

$$A^{[2]} = \text{ReLU}(Z^{[2]})$$

Cache: $(Z^{[2]}, A^{[1]}, W^{[2]}, b^{[2]})$

**Layer 3**:

$$Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}$$

$$A^{[3]} = \sigma(Z^{[3]}) = \hat{Y}$$

Cache: $(Z^{[3]}, A^{[2]}, W^{[3]}, b^{[3]})$

**Compute Loss**:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

### Backward Propagation (Right to Left)

**Initialize**:

$$dA^{[3]} = -\frac{Y}{A^{[3]}} + \frac{1-Y}{1-A^{[3]}}$$

**Layer 3 Backward**:

```python
dA3, cache3 = # from forward prop
dA2, dW3, db3 = backward_propagation_layer_l(dA3, cache3, "sigmoid")
```

Computes: $dW^{[3]}, db^{[3]}, dA^{[2]}$

**Layer 2 Backward**:

```python
dA1, dW2, db2 = backward_propagation_layer_l(dA2, cache2, "relu")
```

Computes: $dW^{[2]}, db^{[2]}, dA^{[1]}$

**Layer 1 Backward**:

```python
dA0, dW1, db1 = backward_propagation_layer_l(dA1, cache1, "relu")
```

Computes: $dW^{[1]}, db^{[1]}, dA^{[0]}$

> **Note**: We don't use $dA^{[0]}$ (gradient w.r.t. input features) for training, so we can discard it.

### Complete Training Iteration

```python
# Forward propagation
A1, cache1 = forward_propagation_layer_l(X, W1, b1, "relu")
A2, cache2 = forward_propagation_layer_l(A1, W2, b2, "relu")
A3, cache3 = forward_propagation_layer_l(A2, W3, b3, "sigmoid")

# Compute loss
cost = compute_cost(A3, Y)

# Initialize backward propagation
dA3 = - (np.divide(Y, A3) - np.divide(1 - Y, 1 - A3))

# Backward propagation
dA2, dW3, db3 = backward_propagation_layer_l(dA3, cache3, "sigmoid")
dA1, dW2, db2 = backward_propagation_layer_l(dA2, cache2, "relu")
dA0, dW1, db1 = backward_propagation_layer_l(dA1, cache1, "relu")

# Update parameters
W3 = W3 - alpha * dW3
b3 = b3 - alpha * db3
W2 = W2 - alpha * dW2
b2 = b2 - alpha * db2
W1 = W1 - alpha * dW1
b1 = b1 - alpha * db1
```

### Visual Flow Diagram

```
FORWARD PROPAGATION:
X ──→ [Layer 1: ReLU] ──→ [Layer 2: ReLU] ──→ [Layer 3: Sigmoid] ──→ Ŷ ──→ Loss
      ↓ cache Z^[1]       ↓ cache Z^[2]       ↓ cache Z^[3]

BACKWARD PROPAGATION:
      dW^[1], db^[1] ←── dW^[2], db^[2] ←── dW^[3], db^[3] ←── dA^[3]
      ↑ use cache         ↑ use cache         ↑ use cache
```

## Summary: Forward and Backward Functions

![Diagram showing forward and backward propagation flow through a 3-layer neural network. The top row shows forward propagation from left to right: Input X flows through ReLU layer with cached z[1], then second ReLU layer with cached z[2], then Sigmoid layer with cached z[3], producing output y-hat. The loss calculation shows L(y-hat, y) equals negative sum over m examples. The bottom row shows backward propagation from right to left with red arrows: Starting from da[3], gradients flow backward through each layer computing dW and db at each stage. Mathematical formulas on the right show: da[3] = -y/a + (1-y)/(1-a), and the backpropagation formula dA[l] equals element-wise product of activation derivative terms. Additional notation shows y-hat as y(m)/a(m) over (1-a(m)). The diagram illustrates the symmetric forward-backward flow of information through the network during training.](/assets/images/deep-learning/neural-networks/week-4/forward_backward_propagation.png)

### Forward Function Summary

| Component | Equation | Purpose |
|-----------|----------|---------|
| **Linear** | $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ | Compute pre-activation |
| **Activation** | $A^{[l]} = g^{[l]}(Z^{[l]})$ | Apply activation function |
| **Cache** | Store $(Z^{[l]}, A^{[l-1]}, W^{[l]}, b^{[l]})$ | Save for backprop |

### Backward Function Summary

| Component | Equation | Purpose |
|-----------|----------|---------|
| **Activation Gradient** | $dZ^{[l]} = dA^{[l]} \odot g'^{[l]}(Z^{[l]})$ | Gradient through activation |
| **Weight Gradient** | $dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$ | Parameter update for $W^{[l]}$ |
| **Bias Gradient** | $db^{[l]} = \frac{1}{m} \sum dZ^{[l]}$ | Parameter update for $b^{[l]}$ |
| **Previous Gradient** | $dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}$ | Pass gradient back |

## Practical Advice

### Don't Worry If It Seems Complex!

> **Important**: If these equations feel abstract or confusing, that's completely normal!

**Why it will become clearer**:

1. **Programming exercise**: Implementing these equations yourself makes them concrete
2. **Working code**: Seeing the equations actually work is enlightening
3. **Practice**: The more you implement, the more intuitive it becomes

### The "Magic" of Deep Learning

Even experienced practitioners are sometimes surprised when deep learning works! Here's why:

**Complexity comes from data, not code**:
- Deep learning code is often just 100-500 lines
- Not 10,000 or 100,000 lines of complex logic
- The **data** does most of the heavy lifting

**The equations are calculus**:
- Forward prop: Chain function compositions
- Backward prop: Chain rule for derivatives
- The derivation is one of the harder ones in machine learning

**It's okay to not derive everything**:
- Focus on **implementing** correctly
- **Understand conceptually** what each step does
- Trust the math (it's been thoroughly verified)

## Key Takeaways

1. **Forward propagation**: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$, then $A^{[l]} = g^{[l]}(Z^{[l]})$
2. **Vectorized forward prop**: Same equations work for all $m$ examples simultaneously
3. **Initialize forward prop**: Start with $A^{[0]} = X$ (input data)
4. **Four backward equations**: $dZ^{[l]}, dW^{[l]}, db^{[l]}, dA^{[l-1]}$ (in that order)
5. **Vectorized backward prop**: Add $\frac{1}{m}$ factor for $dW^{[l]}$ and $db^{[l]}$
6. **Initialize backward prop**: Start with $dA^{[L]} = -\frac{Y}{A^{[L]}} + \frac{1-Y}{1-A^{[L]}}$
7. **Cache is essential**: Store $Z^{[l]}, A^{[l-1]}, W^{[l]}, b^{[l]}$ during forward prop
8. **Element-wise multiplication**: $\odot$ in $dZ^{[l]} = dA^{[l]} \odot g'^{[l]}(Z^{[l]})$
9. **Gradient flow**: Forward goes left-to-right, backward goes right-to-left
10. **Don't need $dA^{[0]}$**: Gradient w.r.t. input features is not used for training
11. **Complete iteration**: Forward → Loss → Backward → Update
12. **NumPy details**: Use `keepdims=True` for `np.sum` to maintain dimensions
13. **Activation functions**: ReLU for hidden layers, sigmoid for binary classification output
14. **The math works**: These are standard calculus equations (chain rule + matrix derivatives)
15. **Practice makes perfect**: Implementing in code makes abstract equations concrete! 