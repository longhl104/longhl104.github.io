---
title: Parameters vs Hyperparameters
parent: Week 4 - Deep Neural Networks
grand_parent: Neural Networks and Deep Learning
nav_order: 7
last_modified_date: 2025-11-24 14:48:00 +1100
---

# Parameters vs Hyperparameters
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

Building effective deep neural networks requires organizing both **parameters** (learned by the model) and **hyperparameters** (chosen by you). Understanding the difference between these two concepts is crucial for efficient model development.

This lesson covers:

- What parameters are (and what they do)
- What hyperparameters are (and why they matter)
- The empirical process of finding good hyperparameters
- Why hyperparameter values change over time
- Practical advice for hyperparameter tuning

> **Key insight**: Hyperparameters control how your parameters are learned, making them "parameters of the learning process" rather than parameters of the model itself.

## Parameters vs Hyperparameters

### What Are Parameters?

**Parameters** are the variables that your model **learns** from the training data:

- **Weights**: $W^{[1]}, W^{[2]}, \ldots, W^{[L]}$
- **Biases**: $b^{[1]}, b^{[2]}, \ldots, b^{[L]}$

These values are:

- **Learned automatically** through gradient descent
- **Updated** during training iterations
- **Optimized** to minimize the cost function

> **Remember**: You don't set these values manually—the learning algorithm finds them!

### What Are Hyperparameters?

**Hyperparameters** are the settings that **you choose** before training that control how the parameters are learned:

#### Common Hyperparameters

| Hyperparameter | Description | Example Values |
|----------------|-------------|----------------|
| **Learning rate** ($\alpha$) | Step size for gradient descent | 0.001, 0.01, 0.1 |
| **Number of iterations** | Training epochs/steps | 1000, 5000, 10000 |
| **Number of layers** ($L$) | Network depth | 2, 3, 5, 10 |
| **Hidden units per layer** ($n^{[l]}$) | Layer width | 50, 100, 256, 512 |
| **Activation functions** | Non-linearity choice | ReLU, tanh, sigmoid |

#### Advanced Hyperparameters (Course 2)

These will be covered in more detail later:

- **Momentum** term
- **Mini-batch size**
- **Regularization** parameters (L2, dropout)
- **Learning rate decay**

### Why "Hyper"parameters?

The prefix "**hyper**" means "above" or "beyond"—hyperparameters sit **above** the regular parameters in the hierarchy:

```
Hyperparameters (α, L, n^[l], ...)
        ↓
  Control how we learn
        ↓
Parameters (W, b)
        ↓
  Determine predictions
        ↓
Output (ŷ)
```

**The relationship**:

- Hyperparameters **control** the learning process
- The learning process **determines** the parameters
- The parameters **determine** the model's predictions

> **Terminology note**: In earlier machine learning eras, people often called $\alpha$ a "parameter." With deep learning's many hyperparameters, we now distinguish them clearly.

## The Empirical Process of Hyperparameter Tuning

### Deep Learning Is Empirical

**Empirical** means you learn from **experiments and observations** rather than pure theory.

> **Reality check**: You cannot know the best hyperparameters in advance—you must try them!

### The Iterative Cycle

```
   ┌─────────────────────────────────────┐
   │                                     │
   ↓                                     │
1. Have an idea                          │
   (e.g., "α = 0.01")                    │
   ↓                                     │
2. Implement and train                   │
   ↓                                     │
3. Observe results                       │
   (cost function behavior)              │
   ↓                                     │
4. Refine idea                           │
   (e.g., "Try α = 0.05")                │
   └──────────────────────────────────────┘
```

### Example: Tuning Learning Rate

Let's say you're experimenting with different learning rates:

#### Experiment 1: $\alpha = 0.01$

```
Cost J
  ↓
  ↓  Gradual decrease
  ↓_______________  (slow but steady)
      Iterations
```

**Observation**: Learning is happening but slowly.

#### Experiment 2: $\alpha = 0.1$ (10× larger)

```
Cost J
  ↑
  ↑  Divergence!
  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑  (cost explodes)
      Iterations
```

**Observation**: Learning rate too large—gradient descent overshoots.

#### Experiment 3: $\alpha = 0.03$

```
Cost J
  ↓
  ↓↓  Fast decrease
  ↓_______ ________  (quick convergence)
      Iterations
```

**Observation**: Good balance—fast learning, stable convergence.

**Decision**: Use $\alpha = 0.03$ for this problem!

### Visualizing Different Hyperparameter Choices

| Learning Rate | Cost Function Behavior | Verdict |
|---------------|------------------------|---------|
| Too small (0.001) | Slow steady decrease | ✓ Works but slow |
| Good (0.01-0.03) | Fast smooth decrease | ✓✓ Ideal |
| Too large (0.1+) | Oscillation or divergence | ✗ Doesn't work |

## Why You Must Experiment

### Reason 1: Cross-Domain Transfer Is Unpredictable

Deep learning is applied across many domains:

- **Computer vision** (images)
- **Speech recognition** (audio)
- **Natural language processing** (text)
- **Structured data** (tables, databases)
- **Recommender systems** (e-commerce)
- **Web search** (ranking)

**The problem**: Hyperparameter intuitions from one domain may not transfer to another!

**Example scenarios**:

| Your Experience | New Problem | What Happens |
|-----------------|-------------|--------------|
| Computer vision expert | Speech recognition | Some intuitions carry over, some don't |
| NLP researcher | Medical imaging | Very different hyperparameters needed |
| E-commerce ML | Financial forecasting | Different data → different settings |

> **Best practice**: Even with experience, always try a range of values when starting a new problem.

### Reason 2: Hyperparameters Change Over Time

Even for the **same problem**, optimal hyperparameters can change:

#### Factors That Cause Changes

1. **Hardware improvements**
   - CPUs get faster → can train with larger batches
   - GPUs improve → can use deeper networks
   - Memory increases → can process more data

2. **Dataset evolution**
   - More data collected → may need more iterations
   - Data distribution shifts → may need different regularization
   - New features added → architecture may need adjustment

3. **Model improvements**
   - Better initialization methods discovered
   - New activation functions developed
   - Advanced optimization algorithms available

#### Practical Recommendation

> **Rule of thumb**: Every few months (or when significant changes occur), re-evaluate your hyperparameters to see if better values exist.

**What to do**:

- Set aside time for hyperparameter exploration
- Try variations around current values
- Compare performance on validation set
- Update if you find improvements

### Building Intuition Over Time

As you work on a problem longer:

**Month 1**: Everything is trial and error

- Try many different values
- Learn what works and what doesn't

**Month 6**: Patterns emerge

- You start to recognize good ranges
- Faster to narrow down choices

**Year 1**: Strong intuition

- Can make educated first guesses
- Still verify with experiments

> **Important**: Intuition is problem-specific! Don't assume it fully transfers to new domains.

## Practical Hyperparameter Tuning Strategy

### Step-by-Step Approach

#### Step 1: Start with Reasonable Defaults

Use commonly recommended starting points:

| Hyperparameter | Starting Value | Reasoning |
|----------------|----------------|-----------|
| Learning rate ($\alpha$) | 0.01 | Works well for many problems |
| Hidden layers ($L$) | 2-3 | Balance complexity and training time |
| Hidden units ($n^{[l]}$) | 50-100 | Enough capacity, not too slow |
| Activation (hidden) | ReLU | Fast, works well in practice |
| Activation (output) | Sigmoid/Softmax | Depends on task (binary/multi-class) |

#### Step 2: Implement and Train

```python
# Example training setup
parameters = initialize_parameters(layer_dims)

for i in range(num_iterations):
    AL, caches = forward_propagation(X, parameters)
    cost = compute_cost(AL, Y)
    grads = backward_propagation(AL, Y, caches)
    parameters = update_parameters(parameters, grads, alpha)
    
    if i % 100 == 0:
        print(f"Cost after iteration {i}: {cost}")
```

#### Step 3: Evaluate on Validation Set

**Don't use training error alone!**

- **Training set**: Used to learn parameters
- **Validation set**: Used to choose hyperparameters
- **Test set**: Used for final evaluation (unbiased)

#### Step 4: Try Variations

Once you have a baseline, experiment systematically:

**Example: Learning rate search**

```
Try: α ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}
Pick: Best validation performance
```

**Example: Architecture search**

```
Try: L ∈ {2, 3, 4, 5}
For each L, try: n^[l] ∈ {50, 100, 200}
Pick: Best validation performance vs training time tradeoff
```

### What to Monitor

While training, watch these indicators:

| Metric | What It Tells You | Action |
|--------|-------------------|--------|
| Training cost decreasing | Learning is happening | ✓ Good |
| Training cost flat | Learning stuck | Try larger $\alpha$ or different architecture |
| Training cost increasing | Divergence | ✗ Reduce $\alpha$ immediately |
| Train/validation gap small | Good generalization | ✓ Model is working |
| Train/validation gap large | Overfitting | Add regularization or get more data |

## Systematic Hyperparameter Search (Preview)

In **Course 2**, you'll learn advanced techniques:

### Grid Search

Try all combinations of hyperparameter values:

```
α = {0.01, 0.03, 0.1}
L = {2, 3, 4}

Total combinations: 3 × 3 = 9 experiments
```

### Random Search

Sample hyperparameter values randomly:

```
α ~ Uniform(0.001, 0.1)
L ~ {2, 3, 4, 5}
n^[l] ~ {50, 100, 200, 500}

Run 20-50 random combinations
```

**Advantage**: Often more efficient than grid search!

### Bayesian Optimization

Use previous results to guide next experiments:

1. Train model with hyperparameters A → score S_A
2. Train model with hyperparameters B → score S_B
3. Use S_A and S_B to intelligently pick next hyperparameters C
4. Repeat

**Advantage**: Finds good hyperparameters with fewer experiments!

## Common Pitfalls to Avoid

### Pitfall 1: Using Test Set for Hyperparameter Tuning

**Wrong approach**:

```
Train on training set
Tune hyperparameters using test set  ← NO!
```

**Why it's wrong**: You're "learning" from test set, so final test performance is overoptimistic.

**Right approach**:

```
Train on training set
Tune hyperparameters using validation set  ← YES!
Final evaluation on test set
```

### Pitfall 2: Not Re-evaluating Hyperparameters

**Wrong**: Set hyperparameters once and never change them.

**Right**: Periodically re-evaluate, especially when:

- You get more data
- Hardware changes
- Problem requirements shift

### Pitfall 3: Overfitting to Validation Set

**Problem**: If you try too many hyperparameter combinations, you might overfit to the validation set itself!

**Solution**:

- Limit the number of experiments
- Use separate validation and test sets
- Consider cross-validation for small datasets

### Pitfall 4: Ignoring Training Time

**Problem**: Best hyperparameters might be too slow to use in practice.

**Solution**: Consider **performance vs training time** tradeoff:

```
Option A: 95% accuracy, trains in 1 hour
Option B: 96% accuracy, trains in 10 hours

Choice depends on your constraints!
```

## The State of Hyperparameter Tuning

### Current Reality (2025)

> **Honest truth**: Hyperparameter tuning remains somewhat of an art, not a perfect science.

**Why it's challenging**:

1. **Problems are diverse**: No universal hyperparameters work everywhere
2. **Infrastructure changes**: CPUs, GPUs, frameworks keep evolving
3. **Data changes**: Distributions shift over time
4. **Research advances**: New techniques constantly emerging

**What this means for you**:

- Experimentation is necessary and expected
- Building intuition takes time and practice
- Even experts experiment and iterate

### Future Directions

Research is making progress on:

- **AutoML**: Automatic hyperparameter optimization
- **Neural Architecture Search (NAS)**: Automatically find network structures
- **Meta-learning**: Learn how to set hyperparameters from past experience
- **Transfer learning**: Reduce need for problem-specific tuning

But for now: **Embrace the empirical process!**

## Practical Advice Summary

### Do's ✓

1. **Start simple**: Begin with 2-3 layers and standard hyperparameters
2. **Use validation set**: Always tune on separate data from training
3. **Try ranges**: Test multiple values (e.g., 0.001, 0.01, 0.1 for $\alpha$)
4. **Monitor metrics**: Watch both training and validation performance
5. **Document everything**: Keep track of what you tried and results
6. **Re-evaluate periodically**: Check if hyperparameters still optimal
7. **Build intuition**: Learn from each experiment

### Don'ts ✗

1. **Don't expect perfect values**: Accept that tuning is empirical
2. **Don't use test set for tuning**: Save it for final evaluation
3. **Don't blindly copy**: Hyperparameters from other domains may not transfer
4. **Don't set once forever**: Optimal values change over time
5. **Don't ignore training time**: Consider practical constraints
6. **Don't try everything**: Be systematic, not exhaustive
7. **Don't get discouraged**: Tuning is hard for everyone!

## Key Takeaways

1. **Parameters** ($W, b$): Learned automatically by the model during training
2. **Hyperparameters** ($\alpha, L, n^{[l]}$, activation): Chosen by you before training
3. **Hyperparameters control parameters**: They determine how learning happens
4. **Deep learning is empirical**: You must try different values and observe results
5. **Iterative process**: Idea → Implement → Evaluate → Refine → Repeat
6. **Learning rate experiments**: Try multiple values and watch cost function behavior
7. **Domain transfer is unpredictable**: Intuitions may or may not carry across problems
8. **Hyperparameters change over time**: Re-evaluate periodically (every few months)
9. **Use validation set**: Tune on separate data from training set
10. **Monitor both metrics**: Training and validation performance tell different stories
11. **Start with defaults**: Use reasonable starting points (e.g., $\alpha = 0.01$)
12. **Be systematic**: Try ranges of values, not random guessing
13. **Consider constraints**: Balance performance vs training time
14. **Build intuition**: Experience helps but doesn't eliminate need for experiments
15. **Embrace uncertainty**: Hyperparameter tuning is hard—even for experts!
