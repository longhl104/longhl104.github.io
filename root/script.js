// Blog posts data structure
const blogPosts = [
    {
        id: 1,
        title: "Getting Started with React Hooks",
        category: "web-development/frontend/react",
        date: "2025-01-15",
        author: "longhl104",
        excerpt: "Learn how to use React Hooks to manage state and side effects in functional components.",
        content: "posts/react-hooks.html"
    },
    {
        id: 2,
        title: "Building RESTful APIs with Node.js",
        category: "web-development/backend/node",
        date: "2025-01-10",
        author: "longhl104",
        excerpt: "A comprehensive guide to creating scalable REST APIs using Express.js and Node.js.",
        content: "posts/nodejs-api.html"
    },
    {
        id: 3,
        title: "Introduction to Machine Learning",
        category: "data-science/machine-learning",
        date: "2025-01-05",
        author: "longhl104",
        excerpt: "Understand the basics of machine learning and how to get started with your first model.",
        content: "posts/ml-intro.html"
    },
    {
        id: 4,
        title: "Docker for Beginners",
        category: "devops/docker",
        date: "2024-12-28",
        author: "longhl104",
        excerpt: "Learn how to containerize your applications with Docker and improve your deployment workflow.",
        content: "posts/docker-beginners.html"
    },
    {
        id: 5,
        title: "Vue 3 Composition API Deep Dive",
        category: "web-development/frontend/vue",
        date: "2024-12-20",
        author: "longhl104",
        excerpt: "Explore the powerful Composition API in Vue 3 and learn how to structure your components better.",
        content: "posts/vue-composition.html"
    },
    {
        id: 6,
        title: "Python Data Analysis with Pandas",
        category: "data-science/data-analysis",
        date: "2024-12-15",
        author: "longhl104",
        excerpt: "Master data manipulation and analysis using the Pandas library in Python.",
        content: "posts/pandas-analysis.html"
    }
];

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeTreeView();
    loadBlogPosts();
    setupEventListeners();
});

// Tree view functionality
function initializeTreeView() {
    const treeToggles = document.querySelectorAll('.tree-toggle');
    
    treeToggles.forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            const parent = toggle.parentElement;
            const nested = parent.querySelector('.tree-nested');
            
            if (nested) {
                nested.classList.toggle('collapsed');
                toggle.classList.toggle('collapsed');
            }
        });
    });

    // Also allow toggling via label click
    const treeLabels = document.querySelectorAll('.tree-label');
    treeLabels.forEach(label => {
        label.addEventListener('click', (e) => {
            e.stopPropagation();
            const parent = label.parentElement;
            const toggle = parent.querySelector('.tree-toggle');
            const nested = parent.querySelector('.tree-nested');
            
            if (nested && toggle) {
                nested.classList.toggle('collapsed');
                toggle.classList.toggle('collapsed');
            }
        });
    });
}

// Load and display blog posts
function loadBlogPosts(filterCategory = null) {
    const blogGrid = document.getElementById('blog-posts');
    blogGrid.innerHTML = '';

    let filteredPosts = blogPosts;
    if (filterCategory) {
        filteredPosts = blogPosts.filter(post => post.category === filterCategory);
    }

    // Sort by date (newest first)
    filteredPosts.sort((a, b) => new Date(b.date) - new Date(a.date));

    if (filteredPosts.length === 0) {
        blogGrid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: #666;">No posts found in this category yet.</p>';
        return;
    }

    filteredPosts.forEach(post => {
        const card = createBlogCard(post);
        blogGrid.appendChild(card);
    });
}

// Create a blog card element
function createBlogCard(post) {
    const card = document.createElement('div');
    card.className = 'blog-card';
    card.onclick = () => loadPost(post.id);

    const categoryName = post.category.split('/').pop().replace(/-/g, ' ');
    const categoryIcon = getCategoryIcon(post.category);

    card.innerHTML = `
        <div class="blog-card-header">
            <h3 class="blog-card-title">${post.title}</h3>
            <div class="blog-card-meta">
                <i class="far fa-calendar"></i> ${formatDate(post.date)}
                <i class="far fa-user" style="margin-left: 1rem;"></i> ${post.author}
            </div>
        </div>
        <div class="blog-card-body">
            <p class="blog-card-excerpt">${post.excerpt}</p>
            <span class="blog-card-category">
                ${categoryIcon} ${categoryName}
            </span>
        </div>
    `;

    return card;
}

// Get icon for category
function getCategoryIcon(category) {
    const iconMap = {
        'react': '<i class="fab fa-react"></i>',
        'vue': '<i class="fab fa-vuejs"></i>',
        'angular': '<i class="fab fa-angular"></i>',
        'node': '<i class="fab fa-node-js"></i>',
        'python': '<i class="fab fa-python"></i>',
        'java': '<i class="fab fa-java"></i>',
        'docker': '<i class="fab fa-docker"></i>',
        'machine-learning': '<i class="fas fa-brain"></i>',
        'data-analysis': '<i class="fas fa-chart-line"></i>',
    };

    for (const [key, icon] of Object.entries(iconMap)) {
        if (category.includes(key)) {
            return icon;
        }
    }
    return '<i class="fas fa-file-alt"></i>';
}

// Format date
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('en-US', options);
}

// Load individual post
async function loadPost(postId) {
    const post = blogPosts.find(p => p.id === postId);
    if (!post) return;

    const homeSection = document.getElementById('home-section');
    const postDetail = document.getElementById('post-detail');
    const postContent = document.getElementById('post-content');

    homeSection.classList.add('hidden');
    postDetail.classList.remove('hidden');

    // Load post content
    try {
        const response = await fetch(post.content);
        if (response.ok) {
            const content = await response.text();
            postContent.innerHTML = `
                <div class="post-meta">
                    <i class="far fa-calendar"></i> ${formatDate(post.date)}
                    <i class="far fa-user" style="margin-left: 1rem;"></i> ${post.author}
                    <i class="fas fa-folder" style="margin-left: 1rem;"></i> ${post.category}
                </div>
                ${content}
            `;
        } else {
            postContent.innerHTML = `
                <h1>${post.title}</h1>
                <div class="post-meta">
                    <i class="far fa-calendar"></i> ${formatDate(post.date)}
                    <i class="far fa-user" style="margin-left: 1rem;"></i> ${post.author}
                </div>
                <p>${post.excerpt}</p>
                <p><em>This is a placeholder. Create the post content at: ${post.content}</em></p>
            `;
        }
    } catch (error) {
        postContent.innerHTML = `
            <h1>${post.title}</h1>
            <div class="post-meta">
                <i class="far fa-calendar"></i> ${formatDate(post.date)}
                <i class="far fa-user" style="margin-left: 1rem;"></i> ${post.author}
            </div>
            <p>${post.excerpt}</p>
            <p><em>This is a placeholder. Create the post content at: ${post.content}</em></p>
        `;
    }

    window.scrollTo(0, 0);
}

// Setup event listeners
function setupEventListeners() {
    // Back button
    document.getElementById('back-btn').addEventListener('click', () => {
        document.getElementById('home-section').classList.remove('hidden');
        document.getElementById('post-detail').classList.add('hidden');
    });

    // Category links
    const categoryLinks = document.querySelectorAll('.tree-link');
    categoryLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const category = link.getAttribute('data-category');
            loadBlogPosts(category);
            
            // Update active state
            categoryLinks.forEach(l => l.style.fontWeight = 'normal');
            link.style.fontWeight = 'bold';

            // Show home section if hidden
            document.getElementById('home-section').classList.remove('hidden');
            document.getElementById('post-detail').classList.add('hidden');

            // Update heading
            const categoryName = category.split('/').pop().replace(/-/g, ' ');
            document.querySelector('#home-section h2').textContent = 
                `Posts in ${categoryName.charAt(0).toUpperCase() + categoryName.slice(1)}`;
        });
    });

    // Navigation links
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            const href = link.getAttribute('href');
            if (href === '#home') {
                document.getElementById('home-section').classList.remove('hidden');
                document.getElementById('post-detail').classList.add('hidden');
                loadBlogPosts();
                document.querySelector('#home-section h2').textContent = 'Latest Posts';
                categoryLinks.forEach(l => l.style.fontWeight = 'normal');
            }
        });
    });
}

// Add a new blog post (for future use)
function addBlogPost(post) {
    blogPosts.unshift(post);
    loadBlogPosts();
}
