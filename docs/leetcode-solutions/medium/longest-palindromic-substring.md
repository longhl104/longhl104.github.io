---
title: 5. Longest Palindromic Substring
parent: Medium Problems
grand_parent: LeetCode Solutions
nav_order: 1
---

# 5. Longest Palindromic Substring
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

![LeetCode problem 5 visualization showing the Expand Around Centers approach with the string b-a-b-a-b-a-d where the center character 'a' is highlighted with a magnifying glass and arrows pointing left and right indicating expansion, with orange highlighting on the left side (a-b) and gray on the right side (b-a-d), demonstrating how palindromes are detected by expanding outward from center positions, set against a dark blue tech-themed background with the C# and LeetCode logos](/assets/images/leetcode/expand_around_centers_visualization.png)

## Problem Description

Given a string `s`, return the longest palindromic substring in `s`.

**Example 1:**

```
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
```

**Example 2:**

```
Input: s = "cbbd"
Output: "bb"
```

**Constraints:**

- `1 <= s.length <= 1000`
- `s` consist of only digits and English letters.

**LeetCode Link:** [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

---

## Approach

This problem can be solved using the **"Expand Around Centers"** technique. The key insight is that every palindrome has a center, and we can check all possible centers.

### Strategy

1. For each possible center in the string, expand outwards while characters match
2. Handle both odd-length palindromes (single character center) and even-length palindromes (between two characters)
3. Track the maximum length found and extract the corresponding substring

### Algorithm Steps

1. **Find maximum length**: For each position, expand around center to find the longest palindrome
2. **Handle two cases**:
   - Odd-length palindromes: center at `i`
   - Even-length palindromes: center between `i` and `i+1`
3. **Extract result**: Once we know the maximum length, scan through the string to find the actual palindrome

---

## Implementation

### C# Solution

```csharp
public class Solution {
    public string LongestPalindrome(string s) {
        var len = 1;
        
        // Find the maximum palindrome length
        for(var i = 0; i < s.Length; ++i) {
            // Check odd-length palindromes (center at i)
            len = Math.Max(len, Expand(s, i, i));
            
            // Check even-length palindromes (center between i and i+1)
            if(i < s.Length - 1) {
                len = Math.Max(len, Expand(s, i, i + 1));
            }
        }

        // Find the actual palindrome with the maximum length
        for(var i = 0; i <= s.Length - len; ++i) {
            if(IsPalindromic(s, i, i + len - 1)) {
                return s.Substring(i, len);
            }
        }

        return "";
    }

    // Check if substring from index i to j is palindromic
    private bool IsPalindromic(string s, int i, int j) {
        while(i <= j) {
            if(s[i] != s[j]) return false;
            ++i;
            --j;
        }
        return true;
    }

    // Expand around center and return the length of palindrome
    private int Expand(string s, int i, int j) {
        var res = 0;
        while(i >= 0 && j < s.Length && s[i] == s[j]) {
            res = j - i + 1;
            --i;
            ++j;
        }
        return res;
    }
}
```

---

## Complexity Analysis

### Time Complexity: O(n²)

- We check each of the `n` positions as potential centers
- For each center, we expand outwards in the worst case `O(n)` times
- The final scan to extract the palindrome is also `O(n)`
- Overall: `O(n²)`

### Space Complexity: O(1)

- We only use a constant amount of extra space for variables
- The input string is not modified
- No additional data structures are needed

---

## Alternative Approaches

### 1. Brute Force - O(n³)

Check every possible substring to see if it's a palindrome.

### 2. Dynamic Programming - O(n²) time, O(n²) space

Use a 2D table to store whether substring `s[i:j]` is a palindrome.

### 3. Manacher's Algorithm - O(n) time

Advanced algorithm that can solve this in linear time, but much more complex to implement.

---

## Test Cases

```csharp
// Test Case 1: Basic example
Input: "babad"
Expected: "bab" or "aba"

// Test Case 2: Even-length palindrome  
Input: "cbbd"
Expected: "bb"

// Test Case 3: Single character
Input: "a"
Expected: "a"

// Test Case 4: No palindrome longer than 1
Input: "abcdef"
Expected: "a" (or any single character)

// Test Case 5: Entire string is palindrome
Input: "racecar"
Expected: "racecar"

// Test Case 6: Multiple palindromes
Input: "abacabad"
Expected: "abacaba"
```

---

## Key Insights

1. **Two types of centers**: Remember to check both odd and even-length palindromes
2. **Expand around center**: This technique is more intuitive than DP for this problem
3. **Early termination**: The solution finds the maximum length first, then locates the actual substring
4. **Edge cases**: Handle single characters and empty strings properly
