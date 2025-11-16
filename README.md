# longhl104.github.io

A modern, tree-structured tech blog built with vanilla HTML, CSS, and JavaScript. This blog features an intuitive hierarchical category system for organizing technical content across various topics including web development, data science, DevOps, mobile development, and security.

## Features

- 🌳 **Tree-like Category Navigation** - Intuitive collapsible sidebar with hierarchical category structure
- 📱 **Responsive Design** - Works seamlessly on desktop, tablet, and mobile devices
- 🎨 **Modern UI** - Clean, professional design with smooth animations and transitions
- 🔍 **Category Filtering** - Filter posts by specific categories or view all posts
- 📝 **Blog Post System** - Easy-to-manage blog posts with metadata (date, author, category)
- 📄 **Markdown Support** - Write posts in Markdown with full formatting support
- 🧮 **LaTeX Math Rendering** - Display mathematical equations using KaTeX
- 💻 **Syntax Highlighting** - Beautiful code blocks with Highlight.js
- 🚀 **Static Site** - Fast loading, no backend required, perfect for GitHub Pages
- 💫 **Interactive Features** - Smooth transitions, hover effects, and dynamic content loading

## Project Structure

```
longhl104.github.io/
├── index.html          # Main HTML file with blog structure
├── styles.css          # CSS styling for responsive design
├── script.js           # JavaScript for interactivity and navigation
├── posts/              # Directory for blog post content
│   ├── react-hooks.html
│   ├── nodejs-api.html
│   ├── ml-intro.html
│   ├── docker-beginners.html
│   ├── vue-composition.html
│   └── pandas-analysis.html
└── README.md
```

## Categories

The blog is organized into the following main categories:

### Web Development
- **Frontend**: React, Vue.js, Angular
- **Backend**: Node.js, Python, Java

### Data Science
- Machine Learning
- Data Analysis
- Data Visualization

### DevOps
- Docker
- Kubernetes
- CI/CD

### Mobile Development
- iOS
- Android
- React Native

### Security
- Web Security
- Cryptography
- Penetration Testing

## Getting Started

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/longhl104/longhl104.github.io.git
cd longhl104.github.io
```

2. Open `index.html` in your browser:
```bash
# On Windows
start index.html

# On macOS
open index.html

# On Linux
xdg-open index.html
```

Or use a local server (recommended):
```bash
# Using Python
python -m http.server 8000

# Using Node.js http-server
npx http-server
```

3. Visit `http://localhost:8000` in your browser

### Deploying to GitHub Pages

1. Push your code to the main branch:
```bash
git add .
git commit -m "Initial blog setup"
git push origin main
```

2. Enable GitHub Pages:
   - Go to your repository settings
   - Navigate to "Pages" section
   - Select "main" branch as the source
   - Save and wait a few minutes

3. Your blog will be live at `https://longhl104.github.io`

## Adding New Blog Posts

You can write blog posts in either **HTML** or **Markdown** format. Markdown posts support LaTeX math equations!

### Option 1: Markdown Post (Recommended)

Create a new Markdown file in the `posts/` directory (e.g., `posts/my-new-post.md`):

```markdown
# Your Post Title

Your introduction paragraph with **bold** and *italic* text.

## Section with Math

The quadratic formula is: $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$

Display equations use double dollar signs:

$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

## Code Examples

\```python
def hello_world():
    print("Hello, World!")
\```

### Lists and More

- Item 1
- Item 2
- Item 3
```

### Option 2: HTML Post

Create a new HTML file in the `posts/` directory (e.g., `posts/my-new-post.html`):

```html
<h1>Your Post Title</h1>

<p>Your introduction paragraph...</p>

<h2>Section Heading</h2>

<p>Your content here...</p>

<pre><code>// Your code examples
function example() {
    console.log('Hello World');
}
</code></pre>
```

### Step 2: Add post metadata

Edit `script.js` and add your post to the `blogPosts` array:

**For Markdown posts:**

```javascript
{
    id: 8,  // Increment the ID
    title: "Your Post Title",
    category: "web-development/frontend/react",  // Use existing category path
    date: "2025-01-20",
    author: "longhl104",
    excerpt: "A brief description of your post...",
    content: "posts/my-new-post.md",
    isMarkdown: true  // Important: Set to true for markdown!
}
```

**For HTML posts:**

```javascript
{
    id: 8,  // Increment the ID
    title: "Your Post Title",
    category: "web-development/frontend/react",
    date: "2025-01-20",
    author: "longhl104",
    excerpt: "A brief description of your post...",
    content: "posts/my-new-post.html",
    isMarkdown: false  // or omit this line
}
```

## Writing LaTeX Math Equations

The blog supports both inline and display math:

### Inline Math

Use single dollar signs for inline equations:

```markdown
The pythagorean theorem is $a^2 + b^2 = c^2$.
```

### Display Math

Use double dollar signs for centered display equations:

```markdown
$$
E = mc^2
$$
```

### Complex Equations

You can write complex mathematical expressions:

```markdown
$$
\mathbf{A} = \begin{bmatrix} 
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
$$
```

For more LaTeX syntax, visit the [KaTeX documentation](https://katex.org/docs/supported.html).

### Step 3: (Optional) Add new categories

If you need a new category, edit the tree structure in `index.html`:

```html
<li>
    <span class="tree-toggle"><i class="fas fa-chevron-down"></i></span>
    <span class="tree-label"><i class="fas fa-icon"></i> Category Name</span>
    <ul class="tree-nested">
        <li><a href="#" class="tree-link" data-category="category-path">
            <i class="fas fa-icon"></i> Subcategory
        </a></li>
    </ul>
</li>
```

## Customization

### Changing Colors

Edit the CSS variables in `styles.css`:

```css
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --accent-color: #e74c3c;
    /* ... other colors */
}
```

### Modifying Layout

- **Sidebar width**: Change `grid-template-columns` in `.layout` class
- **Blog card columns**: Modify `grid-template-columns` in `.blog-grid` class
- **Responsive breakpoints**: Adjust media queries at the bottom of `styles.css`

### Adding Social Links

Edit the footer section in `index.html`:

```html
<div class="social-links">
    <a href="https://github.com/longhl104" target="_blank">
        <i class="fab fa-github"></i>
    </a>
    <!-- Add more social links -->
</div>
```

## Technologies Used

- **HTML5** - Semantic markup
- **CSS3** - Modern styling with Flexbox and Grid
- **JavaScript (ES6+)** - Dynamic functionality
- **Marked.js** - Markdown parsing
- **KaTeX** - LaTeX math rendering
- **Highlight.js** - Syntax highlighting for code blocks
- **Font Awesome** - Icons
- **No frameworks** - Pure vanilla JavaScript for simplicity

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

Feel free to fork this repository and customize it for your own blog. If you find bugs or have suggestions, please open an issue.

## License

This project is open source and available under the MIT License.

## Author

**longhl104**
- GitHub: [@longhl104](https://github.com/longhl104)

## Acknowledgments

- Font Awesome for the icon library
- The open-source community for inspiration

---

Happy blogging! 🚀