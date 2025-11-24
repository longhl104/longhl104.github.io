---
title: Building Blocks of Deep Neural Networks
parent: Week 4 - Deep Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 5
last_modified_date: 2025-11-24 13:29:00 +1100
---

# Building Blocks of Deep Neural Networks

{: .no_toc }
## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

You've already learned the fundamentals of [forward propagation]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-4/forward-propagation-in-a-deep-network.md %}) and [backpropagation]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-3/gradient-descend-for-neural-networks.md %}#backpropagation-equations) from previous lessons. Now let's see how to **modularize** these components to build a complete deep neural network.

The key insight: Think of each layer as a **reusable building block** with:

- A **forward function** (compute activations)
- A **backward function** (compute gradients)
- A **cache** (store values for backpropagation)

> **Design principle**: Building modular, composable functions makes your code clean, debuggable, and reusable!

![Neural network architecture diagram showing forward and backward propagation through layers. Left side shows a multi-layer network with input features x1 through x4 flowing through hidden layers to output ŷ. Annotations indicate layer l with parameters W[l] and b[l], forward propagation with input a[l-1] and output a[l], and backward propagation with gradients da[l] flowing back through cached values z[l]. Right side shows a detailed computation flow diagram for two layers: layer l computes W[l], b[l] from input a[l-1] to produce output a[l] using cache z[l], while layer l-1 below it shows backpropagation computing gradients dW[l], db[l], and da[l-1] from da[l] using cached z[l]. Red arrows indicate backward gradient flow between layers. The diagram illustrates the modular building block concept where each layer performs forward computation and caches values needed for backward gradient computation.](/assets/images/deep-learning/neural-networks/week-4/forward_backward_propagation_diagram.png)

## The Single Layer Building Block

### Focusing on One Layer

Let's zoom in on a single layer $l$ in a deep network and understand what it needs to do.

```
... → [Layer l-1] → [Layer l] → [Layer l+1] → ...
          ↓            ↓            ↓
         a^[l-1]      a^[l]       a^[l+1]
```

**Layer $l$ has**:

- **Parameters**: $W^{[l]}, b^{[l]}$
- **Input**: $a^{[l-1]}$ (activations from previous layer)
- **Output**: $a^{[l]}$ (activations for this layer)

### Forward Propagation: Single Layer

#### The Forward Function

**Function signature**:

```python
def forward_propagation_layer_l(A_prev, W, b, activation):
    """
    Forward propagation for a single layer
    
    Args:
        A_prev: activations from previous layer (a^[l-1])
        W: weight matrix for this layer (W^[l])
        b: bias vector for this layer (b^[l])
        activation: activation function ("relu" or "sigmoid")
    
    Returns:
        A: activations for this layer (a^[l])
        cache: stored values needed for backprop
    """
```

#### Forward Computation Steps

**Step 1: Linear transformation**

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$

**Step 2: Activation function**

$$a^{[l]} = g^{[l]}(z^{[l]})$$

**Step 3: Cache values for backprop**

Store $z^{[l]}$ (and optionally $W^{[l]}, b^{[l]}, a^{[l-1]}$) for later use.

#### Why Cache $z^{[l]}$?

> **Critical**: We need $z^{[l]}$ during backpropagation to compute gradients!

**Reason**: The gradient of the activation function depends on $z^{[l]}$:

$$\frac{\partial a^{[l]}}{\partial z^{[l]}} = g'^{[l]}(z^{[l]})$$

#### Complete Forward Function

```python
def forward_propagation_layer_l(A_prev, W, b, activation):
    """
    Implement forward propagation for a single layer
    """
    # Step 1: Linear transformation
    Z = np.dot(W, A_prev) + b
    
    # Step 2: Activation function
    if activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
    
    # Step 3: Cache values for backprop
    cache = {
        'Z': Z,
        'A_prev': A_prev,
        'W': W,
        'b': b
    }
    
    return A, cache
```

### Backward Propagation: Single Layer

#### The Backward Function

**Function signature**:

```python
def backward_propagation_layer_l(dA, cache, activation):
    """
    Backward propagation for a single layer
    
    Args:
        dA: gradient of cost w.r.t. activations (da^[l])
        cache: values stored during forward prop (Z, A_prev, W, b)
        activation: activation function used ("relu" or "sigmoid")
    
    Returns:
        dA_prev: gradient w.r.t. previous activations (da^[l-1])
        dW: gradient w.r.t. weights (dW^[l])
        db: gradient w.r.t. bias (db^[l])
    """
```

#### Backward Computation Steps

**Given**: $\frac{\partial \mathcal{L}}{\partial a^{[l]}}$ (gradient flowing back from next layer)

**Step 1: Gradient of activation**

$$\frac{\partial \mathcal{L}}{\partial z^{[l]}} = \frac{\partial \mathcal{L}}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial z^{[l]}} = da^{[l]} \cdot g'^{[l]}(z^{[l]})$$

**Step 2: Gradient w.r.t. weights**

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot (a^{[l-1]})^T$$

**Step 3: Gradient w.r.t. bias**

$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}$$

(sum across training examples for vectorized implementation)

**Step 4: Gradient w.r.t. previous activations**

$$\frac{\partial \mathcal{L}}{\partial a^{[l-1]}} = (W^{[l]})^T \cdot \frac{\partial \mathcal{L}}{\partial z^{[l]}}$$

This gradient flows back to the previous layer!

#### Complete Backward Function

```python
def backward_propagation_layer_l(dA, cache, activation):
    """
    Implement backward propagation for a single layer
    """
    # Retrieve cached values
    Z = cache['Z']
    A_prev = cache['A_prev']
    W = cache['W']
    b = cache['b']
    m = A_prev.shape[1]  # number of examples
    
    # Step 1: Gradient of activation function
    if activation == "relu":
        dZ = dA * relu_derivative(Z)
    elif activation == "sigmoid":
        dZ = dA * sigmoid_derivative(Z)
    
    # Step 2: Gradient w.r.t. weights
    dW = (1/m) * np.dot(dZ, A_prev.T)
    
    # Step 3: Gradient w.r.t. bias
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    # Step 4: Gradient w.r.t. previous activations
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db
```

### Summary: Single Layer Functions

#### Forward Function

| Input | Processing | Output |
|-------|-----------|--------|
| $a^{[l-1]}$ | $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$ | $a^{[l]}$ |
| $W^{[l]}, b^{[l]}$ | $a^{[l]} = g^{[l]}(z^{[l]})$ | cache |

**Cache contains**: $z^{[l]}, a^{[l-1]}, W^{[l]}, b^{[l]}$

#### Backward Function

| Input | Processing | Output |
|-------|-----------|--------|
| $da^{[l]}$ | $dz^{[l]} = da^{[l]} \cdot g'^{[l]}(z^{[l]})$ | $da^{[l-1]}$ |
| cache | $dW^{[l]} = \frac{1}{m} dz^{[l]} (a^{[l-1]})^T$ | $dW^{[l]}$ |
|  | $db^{[l]} = \frac{1}{m} \sum dz^{[l]}$ | $db^{[l]}$ |
|  | $da^{[l-1]} = (W^{[l]})^T dz^{[l]}$ |  |

## Building the Complete Deep Network

### Forward Propagation: Full Network

Now let's chain these layer building blocks together!

#### Forward Pass Through All Layers

**Starting point**: $a^{[0]} = X$ (input features)

```
X (a^[0])
   ↓
[Layer 1: forward] → a^[1], cache₁
   ↓
[Layer 2: forward] → a^[2], cache₂
   ↓
[Layer 3: forward] → a^[3], cache₃
   ↓
   ...
   ↓
[Layer L: forward] → a^[L] = ŷ, cache_L
```

**Result**: Predictions $\hat{Y} = a^{[L]}$ and caches for all layers

#### Forward Propagation Algorithm

```python
def L_layer_forward(X, parameters):
    """
    Forward propagation through all L layers
    
    Args:
        X: input data (n^[0], m)
        parameters: dict with W1, b1, W2, b2, ..., WL, bL
    
    Returns:
        AL: final predictions (n^[L], m)
        caches: list of caches for all layers
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers
    
    # Forward through layers 1 to L-1 (ReLU activation)
    for l in range(1, L):
        A_prev = A
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        
        A, cache = forward_propagation_layer_l(A_prev, W, b, "relu")
        caches.append(cache)
    
    # Forward through layer L (sigmoid activation for binary classification)
    WL = parameters[f'W{L}']
    bL = parameters[f'b{L}']
    AL, cache = forward_propagation_layer_l(A, WL, bL, "sigmoid")
    caches.append(cache)
    
    return AL, caches
```

#### What Gets Cached?

For each layer $l = 1, 2, \ldots, L$, we store:

$$\text{cache}_l = \{z^{[l]}, a^{[l-1]}, W^{[l]}, b^{[l]}\}$$

**Why cache all these values?**

| Value | Why Cache It? |
|-------|--------------|
| $z^{[l]}$ | Needed to compute $dz^{[l]} = da^{[l]} \cdot g'^{[l]}(z^{[l]})$ |
| $a^{[l-1]}$ | Needed to compute $dW^{[l]} = dz^{[l]} (a^{[l-1]})^T$ |
| $W^{[l]}$ | Needed to compute $da^{[l-1]} = (W^{[l]})^T dz^{[l]}$ |
| $b^{[l]}$ | Convenient for implementation (optional) |

### Backward Propagation: Full Network

After forward propagation, we have predictions $\hat{Y} = A^{[L]}$. Now we compute gradients!

#### Backward Pass Through All Layers

**Starting point**: $\frac{\partial \mathcal{L}}{\partial A^{[L]}}$ (gradient of loss w.r.t. predictions)

```
dA^[L] (from loss function)
   ↓
[Layer L: backward] → dA^[L-1], dW^[L], db^[L]
   ↓
[Layer L-1: backward] → dA^[L-2], dW^[L-1], db^[L-1]
   ↓
[Layer L-2: backward] → dA^[L-3], dW^[L-2], db^[L-2]
   ↓
   ...
   ↓
[Layer 1: backward] → dA^[0], dW^[1], db^[1]
                        ↑
                   (not used for training)
```

**Result**: Gradients $dW^{[l]}, db^{[l]}$ for all layers

> **Note**: We don't actually need $dA^{[0]}$ (gradient w.r.t. input features) for supervised learning, so we can stop at layer 1.

#### Backward Propagation Algorithm

```python
def L_layer_backward(AL, Y, caches):
    """
    Backward propagation through all L layers
    
    Args:
        AL: final predictions (n^[L], m)
        Y: true labels (n^[L], m)
        caches: list of caches from forward prop
    
    Returns:
        grads: dict with dW1, db1, dW2, db2, ..., dWL, dbL
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Gradient of loss w.r.t. AL (for binary cross-entropy)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Backward through layer L (sigmoid activation)
    current_cache = caches[L-1]
    dA_prev, dW, db = backward_propagation_layer_l(dAL, current_cache, "sigmoid")
    grads[f'dW{L}'] = dW
    grads[f'db{L}'] = db
    
    # Backward through layers L-1 to 1 (ReLU activation)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = backward_propagation_layer_l(dA_prev, current_cache, "relu")
        grads[f'dW{l+1}'] = dW
        grads[f'db{l+1}'] = db
    
    return grads
```

### Complete Network Diagram

#### Forward and Backward Flow

![Neural network architecture diagram showing forward and backward propagation through layers. Left side shows a multi-layer network with input features x1 through x4 flowing through hidden layers to output ŷ. Annotations indicate layer l with parameters W[l] and b[l], forward propagation with input a[l-1] and output a[l], and backward propagation with gradients da[l] flowing back through cached values z[l]. Right side shows a detailed computation flow diagram for two layers: layer l computes W[l], b[l] from input a[l-1] to produce output a[l] using cache z[l], while layer l-1 below it shows backpropagation computing gradients dW[l], db[l], and da[l-1] from da[l] using cached z[l]. Red arrows indicate backward gradient flow between layers. The diagram illustrates the modular building block concept where each layer performs forward computation and caches values needed for backward gradient computation.](/assets/images/deep-learning/neural-networks/week-4/forward-backward-propagation.png)

**Data flow**:

- **Forward**: Activations flow left-to-right, caches stored
- **Backward**: Gradients flow right-to-left, using cached values

## One Training Iteration

### Complete Training Step

One iteration of [gradient descent]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/gradient-descent.md %}) involves:

**Step 1: Forward Propagation**

```python
# Compute predictions
AL, caches = L_layer_forward(X, parameters)
```

**Step 2: Compute Loss**

```python
# Binary cross-entropy loss
cost = compute_cost(AL, Y)
```

**Step 3: Backward Propagation**

```python
# Compute gradients
grads = L_layer_backward(AL, Y, caches)
```

**Step 4: Update Parameters**

```python
# Gradient descent update
for l in range(1, L + 1):
    parameters[f'W{l}'] = parameters[f'W{l}'] - alpha * grads[f'dW{l}']
    parameters[f'b{l}'] = parameters[f'b{l}'] - alpha * grads[f'db{l}']
```

### Training Loop

```python
def train_deep_network(X, Y, layer_dims, learning_rate=0.01, num_iterations=3000):
    """
    Train a deep neural network
    
    Args:
        X: training data (n^[0], m)
        Y: labels (1, m)
        layer_dims: list [n^[0], n^[1], ..., n^[L]]
        learning_rate: alpha for gradient descent
        num_iterations: number of training iterations
    
    Returns:
        parameters: trained weights and biases
    """
    # Initialize parameters
    parameters = initialize_parameters(layer_dims)
    costs = []
    
    # Training loop
    for i in range(num_iterations):
        # Forward propagation
        AL, caches = L_layer_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
        
        # Backward propagation
        grads = L_layer_backward(AL, Y, caches)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print progress
        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")
    
    return parameters, costs
```

## Implementation Details: The Cache

### What Actually Goes in the Cache?

**Conceptually**: We said the cache stores $z^{[l]}$ for backpropagation.

**In practice**: It's convenient to store more!

#### Minimal Cache (Conceptual)

```python
cache = {
    'Z': Z  # Just z^[l]
}
```

This is enough **mathematically**, but you'd need to pass parameters separately.

#### Practical Cache (Implementation)

```python
cache = {
    'Z': Z,          # z^[l] - needed for gradient of activation
    'A_prev': A_prev, # a^[l-1] - needed for dW calculation
    'W': W,          # W^[l] - needed for dA_prev calculation
    'b': b           # b^[l] - convenient for consistency
}
```

**Why include $W^{[l]}$ and $b^{[l]}$?**

- **Convenience**: Don't need to pass parameters separately to backward function
- **Cleaner API**: Backward function is self-contained
- **Minimal overhead**: Storing references to existing arrays

> **Implementation tip**: In your programming exercise, you'll see this extended cache structure—it's just a practical design choice!

### Cache Flow Example

```python
# Forward propagation
def forward_layer(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    A = relu(Z)
    
    # Store everything we might need later
    cache = (Z, A_prev, W, b)
    return A, cache

# Backward propagation
def backward_layer(dA, cache):
    # Unpack cache
    Z, A_prev, W, b = cache
    
    # Now we have everything we need!
    dZ = dA * relu_derivative(Z)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db
```

### Module Summary

Each layer is a **self-contained module**:

| Component | Input | Output | Uses |
|-----------|-------|--------|------|
| **Forward** | $a^{[l-1]}$ | $a^{[l]}$, cache | $W^{[l]}, b^{[l]}$ |
| **Backward** | $da^{[l]}$, cache | $da^{[l-1]}, dW^{[l]}, db^{[l]}$ | cache |
| **Cache** | - | $z^{[l]}, a^{[l-1]}, W^{[l]}, b^{[l]}$ | - |

## What's Next

You now understand the **modular architecture** of deep neural networks! Each layer is a building block with:

- Forward function
- Backward function  
- Cache for communication

In the next lesson, we'll see the **actual implementation** of these building blocks with real code examples.

> **Preview**: You'll implement a complete deep neural network from scratch, putting all these pieces together!

## Key Takeaways

1. **Modular design**: Each layer is a reusable forward-backward block
2. **Forward function**: Takes $a^{[l-1]}$, outputs $a^{[l]}$ and cache
3. **Backward function**: Takes $da^{[l]}$ and cache, outputs $da^{[l-1]}, dW^{[l]}, db^{[l]}$
4. **Cache is crucial**: Stores $z^{[l]}$ (and often $a^{[l-1]}, W^{[l]}, b^{[l]}$) for backprop
5. **Why cache $z^{[l]}$**: Needed to compute activation function gradient $g'^{[l]}(z^{[l]})$
6. **Forward propagation**: Chain layers left-to-right, accumulate caches
7. **Backward propagation**: Chain layers right-to-left, use cached values
8. **Complete iteration**: Forward → Loss → Backward → Update
9. **Parameter updates**: $W^{[l]} \leftarrow W^{[l]} - \alpha \, dW^{[l]}$, $b^{[l]} \leftarrow b^{[l]} - \alpha \, db^{[l]}$
10. **No $da^{[0]}$ needed**: Gradient w.r.t. input features not used in training
11. **Cache contents**: Minimal = $z^{[l]}$; Practical = $z^{[l]}, a^{[l-1]}, W^{[l]}, b^{[l]}$
12. **Self-contained modules**: Forward and backward functions have clean interfaces
13. **Composability**: Stack any number of layers using the same building blocks
14. **Gradient flow**: Activations forward, gradients backward
15. **Implementation convenience**: Extended cache makes backward function self-contained
