#!/bin/bash
set -e

echo "Installing Subsets MCP Server..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)
if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 11 ]); then
    echo "Error: Python 3.11+ required (found $python_version)"
    exit 1
fi

# Install uv if needed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install dependencies
uv sync

# Install the package in editable mode
uv pip install -e .

# Create directories
mkdir -p ~/subsets/{data,catalog}

echo "✓ Installation complete!"
echo ""
echo "Run 'subsets init' to configure your API key."