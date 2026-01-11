#!/usr/bin/env bash
#
# Setup script for reference comparison testing framework
# Creates a Python virtual environment and installs dependencies
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Setting up reference comparison testing framework..."
echo "Directory: $SCRIPT_DIR"
echo ""

# Check for Python 3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    # Check if python is Python 3
    if python --version 2>&1 | grep -q "Python 3"; then
        PYTHON_CMD="python"
    else
        echo "Error: Python 3 is required but not found."
        echo "Please install Python 3.9+ and try again."
        exit 1
    fi
else
    echo "Error: Python is not installed."
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Check Python version is >= 3.9
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo "Error: Python 3.9+ is required (found $PYTHON_VERSION)"
    exit 1
fi

# Check for ImageMagick
if ! command -v magick &> /dev/null; then
    echo "Warning: ImageMagick 'magick' command not found."
    echo "Please install ImageMagick 7.x:"
    echo "  macOS:  brew install imagemagick"
    echo "  Ubuntu: sudo apt-get install imagemagick"
    echo ""
fi

# Create virtual environment if it doesn't exist
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Recreate it? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing virtual environment."
        echo "To update dependencies, activate the venv and run: pip install -r requirements.txt"
        exit 0
    fi
fi

echo "Creating virtual environment..."
$PYTHON_CMD -m venv "$VENV_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then you can run the scripts:"
echo "  python compare_reference.py --help"
echo "  python run_sweep.py --help"
echo "  python analyze_metrics.py --help"
echo "  python suggest_improvements.py --help"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
