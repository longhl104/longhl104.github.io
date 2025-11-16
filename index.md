---
title: Home
layout: home
nav_order: 1
description: "Just the Docs is a responsive Jekyll theme with built-in search that is easily customizable and hosted on GitHub Pages."
permalink: /
---

# Welcome to Long's Blog

This is a place where I share my knowledge and experiences on various topics including programming, technology, and personal development. Feel free to explore the articles and resources available here.

## Latest Posts

{% assign sorted_pages = site.html_pages | where_exp: "page", "page.nav_exclude != true" | sort: "last_modified_at" | reverse %}

{% for page in sorted_pages limit:5 %}
  {% if page.title and page.title != "Home" and page.title != "404" and page.layout != "home" and page.has_children != true %}

- [{{ page.title }}]({{ page.url | relative_url }}) - *{{ page.last_modified_at | date: "%b %e, %Y" }}*
  {% endif %}
{% endfor %}
