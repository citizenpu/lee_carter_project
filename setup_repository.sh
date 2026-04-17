#!/bin/bash

# Script to recreate the Lee Carter project GitHub repository
# This script demonstrates the steps taken to create the repository

echo "Setting up Lee Carter Project repository..."

# Check if source files exist
echo "Checking for source files..."
ls -la "C:/Users/citiz/Downloads/lee_carter_viz.py" "C:/Users/citiz/Downloads/lee_carter_working.py"

# Create project directory
echo "Creating project directory..."
mkdir lee_carter_project
cd lee_carter_project

# Initialize git repository
echo "Initializing git repository..."
git init

# Copy Python files to project directory
echo "Copying Python files..."
cp "C:/Users/citiz/Downloads/lee_carter_viz.py" .
cp "C:/Users/citiz/Downloads/lee_carter_working.py" .

# Verify files were copied
echo "Verifying files..."
ls -la

# Add files to git staging
echo "Adding files to git staging..."
git add .

# Commit files
echo "Committing files..."
git commit -m "Initial commit: Add Lee Carter model Python files

Lee Carter mortality model implementation with visualization.
- lee_carter_working.py: Core model implementation
- lee_carter_viz.py: Visualization tools

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# Create GitHub repository and push
echo "Creating GitHub repository and pushing..."
gh repo create lee_carter_project --public --source=. --remote=origin --push

echo "Repository setup complete!"
echo "Repository URL: https://github.com/citizenpu/lee_carter_project"