// Dark Mode Toggle
;(function () {
  "use strict"

  // Initialize theme on page load
  function initTheme() {
    console.log("🎨 Initializing theme...")
    const savedTheme = localStorage.getItem("theme") || "light"
    console.log("📦 Saved theme:", savedTheme)
    applyTheme(savedTheme, false)
  }

  // Apply theme by enabling/disabling stylesheets
  function applyTheme(theme, log = true) {
    if (log) console.log("🎨 Applying theme:", theme)

    const lightTheme = document.getElementById("light-theme")
    const darkTheme = document.getElementById("dark-theme")

    if (!lightTheme || !darkTheme) {
      console.error("❌ Theme stylesheets not found!")
      return
    }

    if (theme === "dark") {
      lightTheme.disabled = true
      darkTheme.disabled = false
      if (log) console.log("✓ Dark theme enabled")
    } else {
      lightTheme.disabled = false
      darkTheme.disabled = true
      if (log) console.log("✓ Light theme enabled")
    }

    updateIcon(theme)
  }

  // Update theme toggle icon
  function updateIcon(theme) {
    const themeIcon = document.getElementById("theme-icon")
    if (themeIcon) {
      themeIcon.textContent = theme === "dark" ? "☀️" : "🌙"
      console.log("🔄 Icon updated:", themeIcon.textContent)
    }
  }

  // Toggle between light and dark theme
  function toggleTheme() {
    console.log("🖱️ Theme toggle clicked!")
    const currentTheme = localStorage.getItem("theme") || "light"
    const newTheme = currentTheme === "dark" ? "light" : "dark"
    console.log("🔄 Switching from", currentTheme, "to", newTheme)

    localStorage.setItem("theme", newTheme)
    applyTheme(newTheme)
  }

  // Setup theme toggle button
  function setupToggle() {
    const themeToggle = document.getElementById("theme-toggle")

    if (!themeToggle) {
      console.warn("⚠️ Theme toggle button not found")
      return
    }

    themeToggle.addEventListener("click", toggleTheme)
    console.log("✅ Theme toggle ready")
  }

  // Initialize when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      initTheme()
      setupToggle()
    })
  } else {
    initTheme()
    setupToggle()
  }
})()

// Floating Back to Top Button
;(function () {
  "use strict"

  // Create and add floating back to top button
  function createBackToTopButton() {
    const button = document.createElement("button")
    button.id = "floating-back-to-top"
    button.setAttribute("aria-label", "Back to top")
    button.setAttribute("title", "Back to top")
    document.body.appendChild(button)
    return button
  }

  // Show/hide button based on scroll position
  function toggleButtonVisibility(button) {
    const scrollThreshold = 300 // Show button after scrolling 300px
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop

    if (scrollTop > scrollThreshold) {
      button.classList.add("visible")
    } else {
      button.classList.remove("visible")
    }
  }

  // Smooth scroll to top
  function scrollToTop() {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    })
  }

  // Initialize back to top button
  function initBackToTop() {
    const button = createBackToTopButton()

    // Add click event
    button.addEventListener("click", scrollToTop)

    // Show/hide on scroll
    window.addEventListener("scroll", function () {
      toggleButtonVisibility(button)
    })

    // Initial check
    toggleButtonVisibility(button)
  }

  // Initialize when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initBackToTop)
  } else {
    initBackToTop()
  }
})()
