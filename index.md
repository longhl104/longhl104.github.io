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

{% assign all_pages = site.html_pages | where_exp: "page", "page.nav_exclude != true" %}
{% assign pages_with_dates = "" | split: "" %}
{% for page in all_pages %}
  {% assign mod_date = page.last_modified_at | default: page.last_modified_date %}
  {% if mod_date and page.title and page.title != "Home" and page.title != "404" and page.layout != "home" and page.has_children != true %}
    {% assign pages_with_dates = pages_with_dates | push: page %}
  {% endif %}
{% endfor %}
{% assign sorted_pages = pages_with_dates | sort: "last_modified_at" | sort: "last_modified_date" | reverse %}

{% for page in sorted_pages limit:5 %}
  {% assign mod_date = page.last_modified_at | default: page.last_modified_date %}

- [{{ page.title }}]({{ page.url | relative_url }}) - *{{ mod_date | date: "%b %e, %Y" }}*
{% endfor %}
