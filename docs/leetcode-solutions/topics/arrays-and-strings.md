---
title: Arrays and Strings
parent: LeetCode Solutions
nav_order: 4
last_modified_date: 2025-11-29 00:00:00 +0000
---

# Arrays and Strings
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

Arrays and strings are fundamental data structures that appear in many programming problems. This section covers common patterns and techniques for solving array and string manipulation problems.

## Common Patterns

### 1. Two Pointers
Used when you need to compare or process elements from different positions.

**Example Pattern:**
```python
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
```

### 2. Sliding Window
Efficiently process subarrays of a fixed or variable size.

**Example Pattern:**
```python
def max_subarray_sum(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
```

### 3. Hash Map for Lookup
Use dictionaries to achieve O(1) lookups.

**Example Pattern:**
```python
def two_sum(nums, target):
    num_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    
    return []
```

## String-Specific Patterns

### 1. Character Frequency
Count character occurrences for anagram, palindrome problems.

```python
def is_anagram(s, t):
    from collections import Counter
    return Counter(s) == Counter(t)
```

### 2. Palindrome Check
Various techniques for checking palindromes.

```python
def is_palindrome(s):
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    return True
```

## Problem Categories

### Easy Problems
- Two Sum
- Valid Palindrome
- Merge Sorted Array
- Remove Duplicates from Sorted Array

### Medium Problems
- 3Sum
- Container With Most Water
- Longest Substring Without Repeating Characters
- Group Anagrams

### Hard Problems
- Median of Two Sorted Arrays
- Minimum Window Substring
- Longest Valid Parentheses

## Key Techniques

1. **Index Manipulation** - Careful boundary checking
2. **In-place Operations** - Modify arrays without extra space
3. **Prefix/Suffix Processing** - Build cumulative information
4. **Character Encoding** - ASCII values for character operations
5. **String Builder Pattern** - Efficient string concatenation

## Time Complexity Guidelines

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Two Pointers | O(n) | O(1) |
| Sliding Window | O(n) | O(1) |
| Hash Map Lookup | O(n) | O(n) |
| Sorting + Two Pointers | O(n log n) | O(1) |

---

*Specific problems and solutions will be added as they are solved.*