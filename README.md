# Long's Blog

A Jekyll-powered blog with LaTeX support, hosted on GitHub Pages.

## Features

- 📝 Write blog posts in Markdown
- 🔢 Full LaTeX/MathJax support for mathematical equations
- 🎨 Clean, responsive design
- 🚀 Hosted on GitHub Pages

## Local Development

1. Install Ruby and Bundler if you haven't already
2. Clone this repository
3. Run `bundle install` to install dependencies
4. Run `bundle exec jekyll serve` to start the local server
5. Visit `http://localhost:4000` in your browser

## Writing Posts

1. Create a new file in the `_posts` directory
2. Name it using the format: `YYYY-MM-DD-title.md`
3. Add front matter at the top:

```yaml
---
layout: post
title: "Your Post Title"
date: YYYY-MM-DD HH:MM:SS +0000
tags: [tag1, tag2]
---
```

4. Write your content in Markdown

## LaTeX Support

Use LaTeX in your posts:

- Inline math: `$E = mc^2$` renders as $E = mc^2$
- Display math: `$$\int_0^\infty e^{-x}dx = 1$$` renders as a centered equation

## Deployment

Push your changes to the `master` branch on GitHub, and GitHub Pages will automatically build and deploy your site.

## License

Feel free to use this template for your own blog!
