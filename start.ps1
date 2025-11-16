#!/usr/bin/env pwsh
# Simple HTTP server script for running the blog locally

Write-Host "Starting local server for longhl104.github.io..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
}

# Check if Node.js is available
$nodeAvailable = Get-Command node -ErrorAction SilentlyContinue

# Determine which server to use
if ($pythonCmd) {
    Write-Host "Using Python HTTP server..." -ForegroundColor Cyan
    Write-Host "Server running at: http://localhost:8000" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
    Write-Host ""
    
    # Open browser after a short delay
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 2
        Start-Process "http://localhost:8000"
    } | Out-Null
    
    # Start Python server
    & $pythonCmd -m http.server 8000
    
} elseif ($nodeAvailable) {
    Write-Host "Using Node.js http-server (npx)..." -ForegroundColor Cyan
    Write-Host "Server running at: http://localhost:8080" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
    Write-Host ""
    
    # Open browser after a short delay
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 3
        Start-Process "http://localhost:8080"
    } | Out-Null
    
    # Start Node.js server
    npx -y http-server -p 8080
    
} else {
    Write-Host "Error: Neither Python nor Node.js found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install one of the following:" -ForegroundColor Yellow
    Write-Host "  - Python: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "  - Node.js: https://nodejs.org/" -ForegroundColor White
    Write-Host ""
    Write-Host "Alternatively, you can open index.html directly in your browser:" -ForegroundColor Cyan
    Write-Host "  start index.html" -ForegroundColor White
    exit 1
}
