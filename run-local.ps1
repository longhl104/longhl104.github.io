# Jekyll Local Development Server Script
# This script sets up and runs your Jekyll blog locally

Write-Host "Starting Jekyll local development server..." -ForegroundColor Green
Write-Host ""

# Set Ruby path (adjust if your Ruby is installed elsewhere)
$rubyPath = "C:\Ruby34-x64\bin"
if (Test-Path $rubyPath) {
    $env:Path = "$rubyPath;$env:Path"
    Write-Host "✓ Ruby path added: $rubyPath" -ForegroundColor Green
}

# Check if Ruby is installed
Write-Host "Checking Ruby installation..." -ForegroundColor Cyan
try {
    $rubyVersion = ruby --version
    Write-Host "✓ Ruby found: $rubyVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Ruby is not installed or not found!" -ForegroundColor Red
    Write-Host "Expected location: C:\Ruby34-x64\bin" -ForegroundColor Yellow
    Write-Host "Please verify Ruby installation or update the path in this script." -ForegroundColor Yellow
    exit 1
}

# Check if Bundler is installed
Write-Host "Checking Bundler installation..." -ForegroundColor Cyan
try {
    $bundlerVersion = bundle --version
    Write-Host "✓ Bundler found: $bundlerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Bundler is not installed. Installing now..." -ForegroundColor Yellow
    gem install bundler
    Write-Host "✓ Bundler installed successfully" -ForegroundColor Green
}

# Install dependencies if Gemfile.lock doesn't exist or is outdated
if (-not (Test-Path "Gemfile.lock")) {
    Write-Host ""
    Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan
    bundle install
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Checking for dependency updates..." -ForegroundColor Cyan
    bundle check 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Updating dependencies..." -ForegroundColor Cyan
        bundle install
        Write-Host "✓ Dependencies updated successfully" -ForegroundColor Green
    } else {
        Write-Host "✓ All dependencies are up to date" -ForegroundColor Green
    }
}

# Start Jekyll server
Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "Starting Jekyll development server..." -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Your site will be available at: http://localhost:4000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

bundle exec jekyll serve --livereload
