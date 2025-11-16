## What is a Neural Network?

**Deep Learning** refers to training neural networks—sometimes very large neural networks. But what exactly is a neural network?

### Housing Price Prediction Example

![alt text](/posts/deep-learning/deeplearning-ai-course-notes/week1/image.png)

Let's start with a simple example: predicting housing prices. Suppose you have a dataset with six houses where you know:

- The size of each house (in square feet or square meters)
- The price of each house

You want to create a function that predicts a house's price based on its size.

**Traditional Linear Regression Approach:**
If you're familiar with linear regression, you might fit a straight line through the data points. However, there's a problem: eventually, this line would predict negative prices for very small houses, which doesn't make sense.

**A Better Approach:**
Since we know prices can never be negative, we can modify our function to:

1. Stay at zero for very small sizes
2. Then increase linearly as size increases

This creates a "bent" line that starts at zero and then follows a straight path upward. This modified function is actually a very simple neural network—almost the simplest possible!

### The Single Neuron

We can visualize this as a neural network with:

- **Input (x):** Size of the house
- **Neuron:** A single node (represented as a circle) that processes the input
- **Output (y):** Predicted price

**How the neuron works:**

1. Takes the input (size)
2. Computes a linear function
3. Takes the maximum of zero and that linear function
4. Outputs the estimated price

### The ReLU Function

This "maximum of zero" operation is called a **ReLU function** (Rectified Linear Unit):

- **ReLU** stands for **R**ectified **L**inear **U**nit
- "Rectified" simply means "taking the max of 0"
- This creates the characteristic shape: flat at zero, then linear growth
- You'll encounter this function frequently throughout the course

### Building Larger Neural Networks

If a single neuron is like a Lego brick, a larger neural network is built by stacking many of these neurons together. Just as you combine individual Lego bricks to build complex structures, you combine multiple neurons to create more sophisticated neural networks capable of learning complex patterns.
