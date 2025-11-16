# Linear Algebra Basics for Machine Learning

Linear algebra is the foundation of many machine learning algorithms. Understanding vectors, matrices, and their operations is essential for anyone working in data science or AI.

## Vectors and Scalars

A **scalar** is just a single number, while a **vector** is an array of numbers. We can represent a vector as:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

The magnitude (or length) of a vector is calculated using the Euclidean norm:

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

## Matrix Operations

A matrix is a 2D array of numbers. Here's a $3 \times 3$ matrix:

$$\mathbf{A} = \begin{bmatrix} 
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}$$

### Matrix Multiplication

When multiplying matrices $\mathbf{A}$ and $\mathbf{B}$, the element at position $(i,j)$ in the result is:

$$(\mathbf{AB})_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$$

### Transpose

The transpose of a matrix $\mathbf{A}$ is denoted $\mathbf{A}^T$ and is formed by swapping rows and columns:

$$\mathbf{A}^T_{ij} = \mathbf{A}_{ji}$$

## Eigenvalues and Eigenvectors

For a square matrix $\mathbf{A}$, an eigenvector $\mathbf{v}$ and its corresponding eigenvalue $\lambda$ satisfy:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

This equation tells us that multiplying the matrix by the eigenvector only scales the vector by $\lambda$, without changing its direction.

## Example: Computing Eigenvalues

Let's work through a simple $2 \times 2$ example:

$$\mathbf{A} = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

To find eigenvalues, we solve the characteristic equation:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

Expanding:

$$\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = (4-\lambda)(3-\lambda) - 2 = 0$$

$$\lambda^2 - 7\lambda + 10 = 0$$

Solving this quadratic equation gives us $\lambda_1 = 5$ and $\lambda_2 = 2$.

## Python Implementation

Here's how to compute eigenvalues using NumPy:

```python
import numpy as np

# Define the matrix
A = np.array([[4, 1],
              [2, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Verify: A @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"\nVerification for λ={lam:.2f}:")
    print(f"A @ v = {A @ v}")
    print(f"λ * v = {lam * v}")
```

## Applications in Machine Learning

### 1. Principal Component Analysis (PCA)

PCA uses eigenvalue decomposition of the covariance matrix to find the principal components:

$$\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$$

where $\mathbf{X}$ is the centered data matrix.

### 2. Linear Regression

The closed-form solution for linear regression involves matrix operations:

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

where $\mathbf{w}$ are the weights, $\mathbf{X}$ is the feature matrix, and $\mathbf{y}$ is the target vector.

### 3. Neural Networks

The forward pass in a neural network layer is a matrix multiplication:

$$\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

where $\sigma$ is an activation function, $\mathbf{W}$ is the weight matrix, and $\mathbf{b}$ is the bias vector.

## Matrix Calculus

In deep learning, we need to compute gradients. The gradient of a scalar function $f(\mathbf{x})$ with respect to a vector $\mathbf{x}$ is:

$$\nabla_{\mathbf{x}} f = \begin{bmatrix} 
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}$$

## Key Identities

Here are some useful matrix identities:

1. **Distributive**: $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}$
2. **Associative**: $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$
3. **Transpose of product**: $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$
4. **Inverse of product**: $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$

## Norms and Distance Metrics

The **$L^2$ norm** (Euclidean distance) between vectors $\mathbf{x}$ and $\mathbf{y}$ is:

$$\|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

The **$L^1$ norm** (Manhattan distance) is:

$$\|\mathbf{x} - \mathbf{y}\|_1 = \sum_{i=1}^{n}|x_i - y_i|$$

## Conclusion

Linear algebra provides the mathematical foundation for machine learning. Understanding these concepts will help you:

- Implement ML algorithms from scratch
- Debug and optimize existing models
- Read and understand research papers
- Design better neural network architectures

Practice these operations with NumPy, and you'll develop strong intuition for how data transformations work in ML!

## Further Reading

- **Books**: "Linear Algebra and Its Applications" by Gilbert Strang
- **Videos**: MIT OpenCourseWare Linear Algebra course
- **Practice**: Khan Academy and 3Blue1Brown's "Essence of Linear Algebra" series

Remember: Matrix operations are the bread and butter of machine learning. Master them, and you'll understand ML at a deeper level! 🚀
