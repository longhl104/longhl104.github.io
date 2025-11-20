# GitHub Copilot Instructions

## Project Overview

This is a Jekyll-based documentation site using the Just the Docs theme, hosted on GitHub Pages. The site contains Deep Learning course notes from Coursera's Neural Networks and Deep Learning course.

## Content Guidelines

### Writing Style

- **Clarity First**: Transform transcript-style text into well-structured educational content
- **Use Structured Sections**: Break content into clear sections with descriptive headings
- **Include Examples**: Provide code examples, mathematical formulas, and practical demonstrations
- **Add Comparisons**: Use tables to compare approaches (e.g., vectorized vs non-vectorized)
- **Summary Sections**: Include "Key Takeaways" and summary formulas at the end

### Mathematical Notation

- Use LaTeX for all mathematical expressions
- Inline math: `$expression$`
- Block math: `$$expression$$`
- Use proper notation:
  - Vectors: lowercase bold or with arrow ($w$, $x$)
  - Matrices: uppercase ($X$, $W$)
  - Scalars: regular text ($b$, $m$)
  - Superscripts for examples: $x^{(i)}$, $y^{(i)}$
  - Subscripts for features: $x_1$, $w_j$

### Code Examples

- Use Python with NumPy for all code examples
- Include comments explaining key steps
- Show both non-vectorized and vectorized versions when relevant
- Format code blocks with proper syntax highlighting:
  ```python
  # Your code here
  ```

### Links Between Pages

- **Always use Jekyll link syntax** for internal links:
  ```markdown
  [link text]({% link path/to/file.md %})
  ```
- Full path example:
  ```markdown
  [gradient descent]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/gradient-descent.md %})
  ```
- **Link to specific section** (with fragment identifier):
  ```markdown
  [vectorization]({% link docs/deep-learning-ai-course-notes/neural-networks-and-deep-learning/week-2/vectorizing-logistic-regression.md %}#vectorizing-forward-propagation)
  ```
- **Never use** relative links like `gradient-descent.md` or `../file.md`

### Document Structure

Every lesson file should have:

1. **Front Matter**:
   ```yaml
   ---
   title: Page Title
   parent: Week X - Topic
   grand_parent: Course Name
   nav_order: X
   last_modified_date: YYYY-MM-DD HH:MM:SS +TIMEZONE
   ---
   ```

2. **Title and TOC**:
   ```markdown
   # Page Title
   {: .no_toc }

   ## Table of contents
   {: .no_toc .text-delta }

   1. TOC
   {:toc}

   ---
   ```

3. **Main Content** with sections:
   - Introduction
   - Problem statement or motivation
   - Solution or explanation
   - Examples with code
   - Comparisons (tables)
   - What's Next
   - Key Takeaways

### Tables

Use Markdown tables for comparisons:

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

### Visual Elements

- Use blockquotes for important notes:
  ```markdown
  > **Important**: Key insight here
  ```

- Use checkboxes for checklists:
  ```markdown
  - [ ] Item to check
  - [X] Completed item
  ```

## Technical Stack

### Jekyll Configuration

- **Jekyll Version**: 4.x
- **Theme**: Just the Docs
- **Markdown Processor**: Kramdown
- **Math Rendering**: MathJax (configured in layouts)
- **Plugins**:
  - jekyll-last-modified-at
  - jekyll-sitemap
  - jekyll-seo-tag
  - jekyll-github-metadata
  - jekyll-include-cache
  - Custom reading_time.rb plugin

### File Structure

```
docs/
  deep-learning-ai-course-notes/
    neural-networks-and-deep-learning/
      week-1/
      week-2/
        what-is-neural-network.md
        logistic-regression.md
        ...
_includes/
  js/
    custom.js
_layouts/
  default.html
_sass/
  custom/
    custom.scss
_plugins/
  reading_time.rb
```

### Custom Features

1. **Reading Time**: Automatically calculated using custom plugin (200 words/min)
2. **Floating Back-to-Top Button**: Appears after scrolling 300px
3. **Dark Mode Toggle**: Available in custom.js
4. **Custom CSS**: Located in `_sass/custom/custom.scss`

## Code Style

### Python

- Use NumPy conventions
- Prefer vectorized operations over loops
- Include type hints in function definitions when helpful
- Use descriptive variable names:
  - `m` for number of examples
  - `n_x` for number of features
  - `X` for feature matrix (n_x, m)
  - `Y` for label matrix (1, m)
  - `w` for weights
  - `b` for bias
  - `Z` for linear output
  - `A` for activations

### Markdown

- Use ATX-style headers (`#`, `##`, `###`)
- One blank line before and after headers
- One blank line before and after code blocks
- One blank line before and after lists
- No trailing spaces (except for line breaks)
- End files with single newline

## Common Patterns

### Restructuring Transcript to Educational Content

**Before** (transcript style):
```
So in this post we'll talk about... And then you compute... 
And so on and so forth...
```

**After** (structured):
```markdown
## Introduction

Brief overview of what this lesson covers.

## Problem Statement

Clear description of the problem.

## Solution

Step-by-step explanation with formulas.

### Example

```python
# Code example
```

## Key Takeaways

1. Main point 1
2. Main point 2
```

### Vectorization Comparison Pattern

```markdown
### Non-Vectorized (Slow)

```python
for i in range(m):
    # Process each example
```

### Vectorized (Fast)

```python
# Process all examples at once
Z = np.dot(w.T, X) + b
```

| Approach | Speed | Loops |
|----------|-------|-------|
| Non-vectorized | Slow | Multiple |
| Vectorized | Fast | Zero |
```

## Don't Do

- ❌ Use relative markdown links: `[text](file.md)`
- ❌ Use HTML for basic formatting
- ❌ Leave transcript-style text unstructured
- ❌ Skip mathematical notation for formulas
- ❌ Omit code examples
- ❌ Use inline code blocks for multi-line code
- ❌ Forget to link related lessons

## Do

- ✅ Use Jekyll link syntax: `{% link path/to/file.md %}`
- ✅ Structure content with clear sections
- ✅ Use LaTeX for all math: `$x$` and `$$equation$$`
- ✅ Provide code examples with comments
- ✅ Use proper code blocks with syntax highlighting
- ✅ Link related lessons for context
- ✅ Include comparison tables
- ✅ Add "Key Takeaways" sections
- ✅ Show both theory and implementation

## Testing

When making changes:

1. Check markdown lint warnings (cosmetic issues)
2. Verify math renders correctly (MathJax)
3. Test internal links work
4. Ensure code blocks have proper syntax highlighting
5. Restart Jekyll server to see plugin changes: `bundle exec jekyll serve --livereload`

## Resources

- [Just the Docs Documentation](https://just-the-docs.github.io/just-the-docs/)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Kramdown Syntax](https://kramdown.gettalong.org/syntax.html)
- [NumPy Documentation](https://numpy.org/doc/)
