#!/bin/bash
# Script to run the Jekyll site locally with live reload

echo -e "\033[0;32mStarting Jekyll development server...\033[0m"
echo -e "\033[0;36mThe site will be available at http://localhost:4000\033[0m"
echo -e "\033[0;33mPress Ctrl+C to stop the server\033[0m"
echo ""

# Check if bundle is installed
if ! command -v bundle &> /dev/null; then
    echo -e "\033[0;31mError: Bundler is not installed. Please install Ruby and Bundler first.\033[0m"
    exit 1
fi

# Install dependencies if Gemfile.lock doesn't exist
if [ ! -f "Gemfile.lock" ]; then
    echo -e "\033[0;33mInstalling Ruby dependencies...\033[0m"
    bundle install
    if [ $? -ne 0 ]; then
        echo -e "\033[0;31mError: Failed to install dependencies\033[0m"
        exit 1
    fi
fi

# Start Jekyll server with live reload
bundle exec jekyll serve --livereload
