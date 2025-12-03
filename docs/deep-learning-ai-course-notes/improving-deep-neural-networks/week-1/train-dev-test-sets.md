---
title: Train / Dev / Test sets
parent: Week 1 - Practical Aspects of Deep Learning
grand_parent: "Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization"
nav_order: 1
last_modified_date: 2025-11-24 23:09:00 +1100
---

# Train / Dev / Test Sets
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

Properly setting up your training, development (dev), and test sets is one of the most important decisions you'll make when building a neural network. Good data organization can dramatically accelerate your progress and help you find high-performance models faster.

## Why Data Splitting Matters

### The Iterative Nature of Deep Learning

Deep learning is fundamentally an **empirical, iterative process**. When starting a new project, you need to make many decisions:

- **Architecture**: How many layers? How many hidden units per layer?
- **Optimization**: What learning rate? Which optimizer?
- **Regularization**: Dropout rate? L2 penalty?
- **Activation functions**: ReLU? Tanh? Leaky ReLU?

**Key insight**: It's almost impossible to choose the right hyperparameters on your first attempt, even for experienced practitioners.

### The Development Cycle

The typical workflow looks like this:

![Three circular arrows forming a cycle connecting three labeled points: Idea at top, Code at right, and Experiment at left, illustrating the iterative machine learning development process where you generate ideas about model architecture and hyperparameters, implement them in code, run experiments to evaluate performance, and use results to inform new ideas](/assets/images/deep-learning/improving-deep-neural-networks/week-1/iterative_ml_process_diagram.png)

**Your goal**: Make this cycle as fast as possible. Proper data setup is crucial for rapid iteration.

### Domain Expertise Doesn't Always Transfer

Even experienced researchers find that intuitions don't transfer well across domains:

- NLP expert → Computer vision (different best practices)
- Speech recognition → Advertising (different data characteristics)
- Security → Logistics (different problem structures)

**Why?** Optimal choices depend on:

- Amount of available data
- Number of input features
- Hardware configuration (GPU vs CPU, cluster setup)
- Data distribution characteristics
- Application-specific constraints

## The Three Data Sets

### Purpose of Each Set

| Set | Purpose | Usage |
|-----|---------|-------|
| **Training Set** | Learn parameters ($w$, $b$) | Train your model repeatedly |
| **Dev Set** | Compare model architectures | Choose between different models |
| **Test Set** | Unbiased performance estimate | Final evaluation only |

> **Note**: The dev set is also called the **hold-out cross-validation set** or **validation set**. These terms are interchangeable.

### The Standard Workflow

1. **Train**: Train multiple models on the training set
2. **Validate**: Evaluate all models on the dev set
3. **Select**: Pick the best-performing model from dev set results
4. **Test**: Evaluate the final selected model on the test set (once!)

The test set gives you an **unbiased estimate** because you haven't used it to make any decisions about your model.

## Data Split Ratios: Then vs Now

### Traditional Machine Learning Era (Small Data)

**Dataset size**: 100 to 10,000 examples

**Common splits**:

- 70% train / 30% test
- 60% train / 20% dev / 20% test

These ratios were reasonable when data was limited.

### Modern Deep Learning Era (Big Data)

**Dataset size**: 1 million+ examples

**Modern splits**:

- 1,000,000 examples: 98% train / 1% dev / 1% test (10K dev, 10K test)
- 10,000,000 examples: 99.5% train / 0.25% dev / 0.25% test
- 100,000,000 examples: 99.9% train / 0.05% dev / 0.05% test

### Why Such Small Dev/Test Sets?

**Dev set purpose**: Determine which of several models performs best

- Do you need 200,000 examples to compare 10 algorithms? **No!**
- 10,000 examples is usually more than sufficient
- You just need enough data to distinguish between models with statistical confidence

**Test set purpose**: Estimate final model performance

- Again, 10,000-50,000 examples is typically enough
- You need enough data for a confident performance estimate, not 20% of your data

### General Guidelines by Dataset Size

| Total Examples | Train | Dev | Test | Reasoning |
|---------------|-------|-----|------|-----------|
| 100-1,000 | 60% | 20% | 20% | Traditional splits work fine |
| 10,000 | 60% | 20% | 20% | Still reasonable |
| 100,000 | 90% | 5% | 5% | Starting to shift |
| 1,000,000 | 98% | 1% | 1% | Modern big data approach |
| 10,000,000+ | 99%+ | <1% | <1% | Maximize training data |

## Mismatched Train and Test Distributions

### The Modern Reality

In modern deep learning, it's increasingly common to have **different distributions** for training vs dev/test sets.

### Example: Cat Photo App

**Scenario**: Building an app to find cat pictures for users

**Data sources**:

- **Training set**: High-quality cat photos scraped from the Internet
  - Professional photography
  - High resolution
  - Perfect framing and lighting
  - Large volume available (millions of images)

- **Dev/Test sets**: Cat photos uploaded by users
  - Phone camera quality
  - Lower resolution
  - Casual conditions (blurry, poor lighting)
  - Limited volume available (thousands of images)

### Critical Rule for Mismatched Distributions

> **Rule**: Dev and test sets must come from the same distribution!

**Why this matters**:

$$
\text{Train distribution} \neq \text{Dev/Test distribution} \text{ (OK)}
$$

$$
\text{Dev distribution} \neq \text{Test distribution} \text{ (NOT OK)}
$$

**Reasoning**:

1. You'll spend significant time optimizing performance on the dev set
2. If dev and test distributions differ, good dev performance doesn't guarantee good test performance
3. You want to "aim" at the same target you'll be evaluated on

### When to Use Mismatched Distributions

Use this approach when:

- ✅ You can easily acquire large training data from alternative sources
- ✅ Your true target data is limited
- ✅ Alternative data is relevant to your task

**Example scenarios**:

- Medical imaging: Public datasets + small hospital-specific data
- Speech recognition: Audiobooks + real user recordings
- Product recommendations: Historical data + recent user behavior

## When You Don't Need a Test Set

### Test Set is Optional

The test set serves **one purpose**: Give an unbiased estimate of your final model's performance.

**If you don't need that unbiased estimate**, you can skip the test set!

### Train + Dev Only (No Test)

**Workflow**:

1. Train on training set
2. Evaluate multiple models on dev set
3. Pick the best model based on dev set
4. Deploy that model

**Caveat**: Because you've optimized for the dev set, your performance estimate is **biased** (overly optimistic).

**When this is acceptable**:

- Rapid prototyping and experimentation
- Internal tools where slight performance overestimation is acceptable
- Situations where you can quickly observe real-world performance

### Terminology Warning

⚠️ **Common confusion**: Many teams call their dev set a "test set" when they don't have a separate test set.

**What they actually have**:

- Training set
- Dev set (mislabeled as "test set")
- No true test set

**Problem**: They're overfitting to what they call the "test set", which defeats the purpose of unbiased evaluation.

**Better terminology**: Just be honest and call it a dev set!

## Practical Guidelines

### Quick Reference: Setting Up Your Data

1. **Split your data** into train/dev/test
2. **Choose split ratios** based on total data size
3. **Ensure dev and test** come from the same distribution
4. **Make dev/test** represent your target application
5. **Consider skipping test set** if you don't need unbiased estimates

### Checklist for Data Setup

- [ ] Do I have enough data for each split to be meaningful?
- [ ] Are my dev and test sets from the same distribution?
- [ ] Do my dev/test sets represent real-world usage?
- [ ] Is my dev set large enough to distinguish between models?
- [ ] Is my test set large enough for confident performance estimation?
- [ ] Have I considered whether I actually need a test set?

### Red Flags to Avoid

❌ **Don't**:

- Use different distributions for dev and test sets
- Make dev/test sets too large when you have big data
- Use the test set multiple times to make decisions
- Mix real target data into training when it should be in dev/test

✅ **Do**:

- Put real target data in dev/test sets
- Use alternative sources to bulk up training data
- Keep dev/test distributions identical
- Reserve test set for final evaluation only

## Key Takeaways

1. **Data organization matters**: Proper train/dev/test setup accelerates progress significantly
2. **Deep learning is iterative**: You'll cycle through many experiments before finding good hyperparameters
3. **Traditional splits are outdated**: With big data, use much smaller dev/test sets (1-2% each)
4. **Dev set size**: Just needs to distinguish between models (10K examples often sufficient)
5. **Test set size**: Just needs confident performance estimate (10K-50K examples often sufficient)
6. **Distribution matching**: Dev and test must have same distribution
7. **Distribution mismatch**: Train can differ from dev/test if it gives you more data
8. **Test set optional**: Skip it if you don't need unbiased performance estimates
9. **Terminology confusion**: Many call dev sets "test sets" - be aware of this
10. **Domain transfer is hard**: Intuitions from one domain rarely transfer to others
11. **Hardware matters**: GPU/CPU configuration affects optimal hyperparameters
12. **Empirical process**: You must experiment - no one gets it right the first time
13. **Fast iteration wins**: Efficient data setup enables rapid experimentation
14. **Aim at the right target**: Put representative data in dev/test, not training
15. **More data usually helps**: Use creative tactics to acquire more training data
