# Markdown & LaTeX Support - Feature Summary

Your blog now supports **Markdown posts with LaTeX math equations**! 🎉

## What's New

### 1. Libraries Added
- **Marked.js** - Converts Markdown to HTML
- **KaTeX** - Renders LaTeX math equations beautifully
- **Highlight.js** - Syntax highlighting for code blocks

### 2. Updated Files
- `index.html` - Added CDN links for the libraries
- `script.js` - Added markdown parsing and math rendering logic
- `styles.css` - Enhanced code block and math equation styling
- `README.md` - Added documentation for markdown and LaTeX usage

### 3. New Content
- **Sample Post**: `posts/linear-algebra-basics.md` - Demonstrates markdown with LaTeX
- **Guide**: `MARKDOWN_GUIDE.md` - Complete reference for writing posts

## How to Use

### Writing a Markdown Post

1. Create a `.md` file in the `posts/` directory
2. Write your content using markdown syntax
3. Add LaTeX equations using `$` (inline) or `$$` (display)
4. Add the post to `script.js` with `isMarkdown: true`

### Example Post Entry

```javascript
{
    id: 7,
    title: "My Amazing Post",
    category: "data-science/machine-learning",
    date: "2025-01-20",
    author: "longhl104",
    excerpt: "Learn about cool math stuff!",
    content: "posts/my-post.md",
    isMarkdown: true  // Important!
}
```

## Math Examples

### Inline Math
```markdown
The formula is $E = mc^2$ and the quadratic is $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$.
```

### Display Math
```markdown
$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$
```

### Matrices
```markdown
$$
\mathbf{A} = \begin{bmatrix} 
1 & 2 \\
3 & 4
\end{bmatrix}
$$
```

## Code Blocks with Syntax Highlighting

````markdown
```python
def hello_world():
    print("Hello, World!")
```

```javascript
const greeting = "Hello, World!";
console.log(greeting);
```
````

## Testing Your Blog

1. Open `index.html` in a browser
2. Click on "Linear Algebra Basics for Machine Learning"
3. Verify:
   - Markdown is rendered as HTML
   - Math equations display correctly
   - Code blocks have syntax highlighting

## Supported Languages for Code Highlighting

- Python
- JavaScript
- Java
- C/C++
- Go
- Rust
- Ruby
- PHP
- SQL
- Bash/Shell
- And many more!

## Tips

1. **Preview locally** before pushing to GitHub
2. **Use display math** for important equations
3. **Test math rendering** - some LaTeX commands may not be supported by KaTeX
4. **Keep markdown simple** - complex HTML inside markdown may not render correctly
5. **Use code fences** with language specification for best highlighting

## What Works

✅ Standard markdown syntax (headings, lists, links, etc.)
✅ Inline and display math equations
✅ Code blocks with syntax highlighting
✅ Tables, blockquotes, and horizontal rules
✅ Most KaTeX/LaTeX commands
✅ GitHub Flavored Markdown features

## References

- **Markdown Guide**: See `MARKDOWN_GUIDE.md`
- **KaTeX Docs**: https://katex.org/docs/supported.html
- **Marked.js**: https://marked.js.org/
- **Highlight.js**: https://highlightjs.org/

---

Your blog is now ready for technical writing with beautiful math equations! 🚀📊✨
