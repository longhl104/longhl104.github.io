#!/usr/bin/env pwsh
# Script to run the Jekyll site locally with live reload

Write-Host "Starting Jekyll development server..." -ForegroundColor Green
Write-Host "The site will be available at http://localhost:4000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Check if bundle is installed
if (!(Get-Command bundle -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Bundler is not installed. Please install Ruby and Bundler first." -ForegroundColor Red
    exit 1
}

# Install dependencies if Gemfile.lock doesn't exist
if (!(Test-Path "Gemfile.lock")) {
    Write-Host "Installing Ruby dependencies..." -ForegroundColor Yellow
    bundle install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

# Start Jekyll server with live reload
bundle exec jekyll serve --livereload
