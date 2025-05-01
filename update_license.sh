#!/bin/bash

# Script to update license headers in all Python source files

# Create license header from LICENSE file
LICENSE_TEXT=$(cat LICENSE)
LICENSE_HEADER="# $(echo "$LICENSE_TEXT" | sed 's/^/# /g' | sed 's/^# $/# /g')"

# Find all Python files in the repository (excluding .git, __pycache__, etc.)
find . -type f -name "*.py" \
    -not -path "*/\.*" \
    -not -path "*/__pycache__*" \
    -not -path "*/build/*" \
    -not -path "*/dist/*" \
    -not -path "*/venv/*" \
    -not -path "*/env/*" \
    | while read file; do
    
    echo "Processing $file"
    
    # Check if file already has a license header
    if grep -q "Copyright (c) 2024 NeuralMag team" "$file"; then
        echo "  License already exists, updating..."
        # Remove existing license header (up to the first blank line after copyright)
        sed -i '1,/^$/d' "$file"
    fi
    
    # Create a temporary file with license and original content
    (echo "$LICENSE_HEADER"; echo ""; cat "$file") > temp_file
    mv temp_file "$file"
    
    echo "  Done"
done

echo "License update complete!"
#!/bin/bash

# Script to update license headers in all Python source files

# Create a simplified license header for Python files
LICENSE_HEADER="# MIT License
#
# Copyright (c) 2024 NeuralMag team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the \"Software\"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE."

# Find all Python files in the repository (excluding .git, __pycache__, etc.)
find . -type f -name "*.py" \
    -not -path "*/\.*" \
    -not -path "*/__pycache__*" \
    -not -path "*/build/*" \
    -not -path "*/dist/*" \
    -not -path "*/venv/*" \
    -not -path "*/env/*" \
    | while read file; do
    
    echo "Processing $file"
    
    # Check if file already has a license header
    if grep -q "Copyright (c) 2024 NeuralMag team" "$file"; then
        echo "  License already exists, updating..."
        # Find the line number where the existing license header ends
        header_end=$(grep -n "^$" "$file" | head -1 | cut -d: -f1)
        if [ -z "$header_end" ]; then
            header_end=1
        fi
        
        # Remove existing license header
        sed -i "1,${header_end}d" "$file"
    fi
    
    # Create a temporary file with license and original content
    (echo "$LICENSE_HEADER"; echo ""; cat "$file") > "${file}.tmp"
    mv "${file}.tmp" "$file"
    
    echo "  Done"
done

echo "License update complete!"
