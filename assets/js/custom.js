// Dark Mode Toggle
(function ()
{
  "use strict";

  // Initialize theme on page load
  function initTheme()
  {
    const savedTheme = localStorage.getItem("theme") || "light";
    applyTheme(savedTheme);
  }

  // Apply theme by enabling/disabling stylesheets
  function applyTheme(theme)
  {
    const lightTheme = document.getElementById("light-theme");
    const darkTheme = document.getElementById("dark-theme");

    if (!lightTheme || !darkTheme)
    {
      return;
    }

    if (theme === "dark")
    {
      lightTheme.disabled = true;
      darkTheme.disabled = false;
    } else
    {
      lightTheme.disabled = false;
      darkTheme.disabled = true;
    }

    updateIcon(theme);
  }

  // Update theme toggle icon
  function updateIcon(theme)
  {
    const themeIcon = document.getElementById("theme-icon");
    if (themeIcon)
    {
      themeIcon.textContent = theme === "dark" ? "☀️" : "🌙";
    }
  }

  // Toggle between light and dark theme
  function toggleTheme()
  {
    const currentTheme = localStorage.getItem("theme") || "light";
    const newTheme = currentTheme === "dark" ? "light" : "dark";

    localStorage.setItem("theme", newTheme);
    applyTheme(newTheme);
  }

  // Setup theme toggle button
  function setupToggle()
  {
    const themeToggle = document.getElementById("theme-toggle");

    if (!themeToggle)
    {
      return;
    }

    themeToggle.addEventListener("click", toggleTheme);
  }

  // Initialize when DOM is ready
  if (document.readyState === "loading")
  {
    document.addEventListener("DOMContentLoaded", function ()
    {
      initTheme();
      setupToggle();
    });
  } else
  {
    initTheme();
    setupToggle();
  }
})();
