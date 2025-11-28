---
title: Medium Problems
parent: LeetCode Solutions
nav_order: 2
has_children: true
---

# Medium Problems
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Medium problems require a deeper understanding of algorithms and data structures. They often involve:

- Multiple solution approaches with trade-offs
- Optimization challenges
- Complex state management
- Combined algorithmic techniques

## Problem Categories

### Dynamic Programming

- 1D and 2D DP problems
- Optimization problems
- State transition modeling

### Trees & Graphs

- Binary tree operations
- Graph traversals (BFS/DFS)
- Tree construction and modification

### Advanced Arrays

- Sliding window techniques
- Two-pointer variations
- Subarray problems

### Backtracking

- Permutations and combinations
- Constraint satisfaction
- Search space pruning

## Solved Problems

*Problems will be added as they are solved*

## Key Patterns for Medium Problems

### 1. Dynamic Programming

```python
def dp_example(nums):
    dp = [0] * len(nums)
    dp[0] = nums[0]
    
    for i in range(1, len(nums)):
        dp[i] = max(dp[i-1] + nums[i], nums[i])
    
    return max(dp)
```

### 2. Sliding Window

```python
def sliding_window_example(s, k):
    left = 0
    max_len = 0
    char_count = {}
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

### 3. Tree Traversal

```python
def tree_traversal_example(root):
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
```

## Tips for Medium Problems

1. **Multiple approaches** - Consider brute force, optimized, and alternative solutions
2. **Time vs Space** - Understand complexity trade-offs
3. **Edge cases** - More complex boundary conditions
4. **State management** - Track multiple variables or states
5. **Pattern combination** - Often combine multiple algorithmic patterns

---

*Solutions will be added as problems are solved and documented.*
