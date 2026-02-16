#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the project root (assuming script is in root)
cd "$SCRIPT_DIR"

echo "Starting SPH Analysis Suite..."
echo "--------------------------------"

# Check if python3 is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

# Run the analysis orchestration script
$PYTHON_CMD src/run_analysis.py

echo "--------------------------------"
echo "Analysis finished."
