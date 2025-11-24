---
title: Week 1 - Practical Aspects of Deep Learning
parent: Improving Deep Neural Networks
grand_parent: DeepLearning.AI Course Notes
nav_order: 1
has_children: true
---

# Week 1: Practical Aspects of Deep Learning

{: .no_toc }

## Overview

Week 1 focuses on the **practical techniques** you need to build deep learning models that work well in real applications. You'll learn how to set up your data, diagnose problems, and apply regularization to improve performance.

## Learning Objectives

By the end of this week, you'll be able to:

- Split data into train/dev/test sets effectively
- Diagnose whether you have high bias or high variance
- Apply L2 regularization and dropout to prevent overfitting
- Understand and prevent vanishing/exploding gradients
- Initialize weights properly for faster training
- Verify your backpropagation implementation with gradient checking

## Topics Covered

### Setting Up Your Machine Learning Application

- Train/dev/test set splits
- When to use different split ratios
- Ensuring dev and test sets come from the same distribution
- When it's okay not to have a test set

### Bias and Variance

- Understanding the bias-variance tradeoff
- Diagnosing high bias (underfitting)
- Diagnosing high variance (overfitting)
- Basic recipe for machine learning

### Regularization

- L2 regularization (weight decay)
- Dropout regularization
- Other regularization techniques: data augmentation, early stopping
- Why regularization reduces overfitting

### Setting Up Your Optimization Problem

- Vanishing and exploding gradients
- Weight initialization strategies: Xavier, He initialization
- Gradient checking for debugging backpropagation
- When to use gradient checking (and when not to)

## Programming Assignment

**Assignment**: Practical aspects of deep learning

You'll implement:

- Regularization techniques (L2 and dropout)
- Gradient checking to verify backpropagation
- Initialization schemes to improve training

## Key Takeaways

- **Data setup matters**: Proper train/dev/test splits are crucial
- **Diagnose first**: Identify whether you have bias or variance problems
- **Regularization is essential**: Always use it for production models
- **Initialization matters**: Good initialization speeds up training significantly
- **Verify your code**: Gradient checking catches subtle bugs
