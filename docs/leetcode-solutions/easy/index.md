---
title: Easy Problems
parent: LeetCode Solutions
nav_order: 1
has_children: true
permalink: /docs/leetcode-solutions/easy/
last_modified_date: 2025-11-29 00:00:00 +0000
---

# Easy Problems
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Easy problems focus on fundamental programming concepts and basic algorithm patterns. These problems are excellent for:

- Building confidence with coding interviews
- Learning essential data structures
- Understanding basic algorithmic thinking
- Practicing clean code implementation

## Problem Categories

### Arrays & Strings
- Two-pointer techniques
- Array manipulation
- String processing

### Math & Logic
- Basic mathematical operations
- Bit manipulation
- Number theory problems

### Data Structures
- Hash tables (dictionaries)
- Sets and basic operations
- Stack and queue basics

## Solved Problems

*Problems will be added as they are solved*

## Key Patterns for Easy Problems

### 1. Two Pointers
```python
def two_pointer_example(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        # Process arr[left] and arr[right]
        left += 1
        right -= 1
```

### 2. Hash Table Lookup
```python
def hash_table_example(arr, target):
    seen = {}
    for i, num in enumerate(arr):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

### 3. Single Pass
```python
def single_pass_example(arr):
    result = 0
    for num in arr:
        result += num  # Process in one pass
    return result
```

## Tips for Easy Problems

1. **Read carefully** - Understand constraints and edge cases
2. **Start simple** - Brute force first, optimize later
3. **Test thoroughly** - Consider empty inputs, single elements
4. **Clean code** - Easy problems are about implementation clarity
5. **Pattern recognition** - Many easy problems reuse similar patterns

---

*Solutions will be added as problems are solved and documented.*