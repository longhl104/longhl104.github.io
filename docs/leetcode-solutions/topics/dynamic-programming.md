---
title: Dynamic Programming
parent: LeetCode Solutions
nav_order: 5
last_modified_date: 2025-11-29 00:00:00 +0000
---

# Dynamic Programming
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Dynamic Programming (DP) is an optimization technique that solves complex problems by breaking them into smaller subproblems. It's particularly effective when subproblems overlap and optimal substructure exists.

## Core Concepts

### 1. Optimal Substructure
A problem has optimal substructure if an optimal solution can be constructed from optimal solutions of its subproblems.

### 2. Overlapping Subproblems
The same subproblems are solved multiple times in a naive recursive approach.

### 3. Memoization vs Tabulation
- **Memoization**: Top-down approach with recursion + cache
- **Tabulation**: Bottom-up approach filling a table

## Common DP Patterns

### 1. Linear DP (1D)
Problems where each state depends on previous states in a linear fashion.

**Example - House Robber:**
```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[-1]
```

### 2. Grid DP (2D)
Problems involving 2D grids or matrices.

**Example - Unique Paths:**
```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]
```

### 3. Knapsack Pattern
Choose items to maximize value while staying within weight constraint.

**Example - 0/1 Knapsack:**
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # Don't take item
                    dp[i-1][w - weights[i-1]] + values[i-1]  # Take item
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

### 4. Subsequence DP
Finding optimal subsequences (LIS, LCS, etc.).

**Example - Longest Increasing Subsequence:**
```python
def length_of_LIS(nums):
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)
```

## Problem Categories by Difficulty

### Easy Problems
- Climbing Stairs
- House Robber
- Maximum Subarray
- Best Time to Buy and Sell Stock

### Medium Problems
- Coin Change
- Longest Increasing Subsequence
- Unique Paths
- Word Break

### Hard Problems
- Edit Distance
- Regular Expression Matching
- Longest Common Subsequence
- Maximum Rectangle

## DP Optimization Techniques

### 1. Space Optimization
Reduce space complexity by only keeping necessary previous states.

```python
def rob_optimized(nums):
    prev2 = prev1 = 0
    
    for num in nums:
        current = max(prev1, prev2 + num)
        prev2 = prev1
        prev1 = current
    
    return prev1
```

### 2. Rolling Array
Use modular arithmetic to cycle through array indices.

```python
def unique_paths_optimized(m, n):
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    
    return dp[n-1]
```

## Problem-Solving Framework

### Step 1: Identify DP Problem
Look for:
- Optimal substructure
- Overlapping subproblems
- Optimization (min/max) or counting problems

### Step 2: Define State
- What does dp[i] represent?
- What are the dimensions?
- What are the base cases?

### Step 3: State Transition
- How do you compute dp[i] from previous states?
- What are the recurrence relations?

### Step 4: Implementation
- Choose between memoization or tabulation
- Handle base cases carefully
- Consider space optimization

## Common Mistakes

1. **Incorrect base cases** - Always verify edge conditions
2. **Wrong state definition** - Ensure state captures all necessary information
3. **Off-by-one errors** - Careful with array indexing
4. **Missing optimization** - Consider space complexity improvements

---

*Specific DP problems and solutions will be added as they are solved.*