#!/bin/bash

# Script to update license headers in all Python source files in neuralmag directory

# Create a license header for Python files
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

# Find all Python files in the neuralmag directory
find ./neuralmag -type f -name "*.py" \
    -not -path "*/__pycache__*" \
    | while read file; do
    
    echo "Processing $file"
    
    # Create a temporary file for the processed content
    temp_file="${file}.tmp"
    
    # Check if file has a shebang line
    has_shebang=0
    if head -1 "$file" | grep -q "^#!"; then
        has_shebang=1
        shebang_line=$(head -1 "$file")
    fi
    
    # Use sed to remove license headers
    # This handles both comment-style and docstring-style headers
    
    # First, create a clean file without any license header
    if grep -q "Copyright" "$file"; then
        echo "  Removing existing license header..."
        
        # Create a temporary Python script to handle the removal
        python_script=$(mktemp)
        cat > "$python_script" << 'EOF'
import re
import sys

def remove_license_header(content):
    # Check for docstring license
    docstring_match = re.match(r'("""|\'\'\')(.*?)("""|\'\'\')', content, re.DOTALL)
    if docstring_match and ('Copyright' in docstring_match.group(2) or 'License' in docstring_match.group(2)):
        # Remove the docstring license
        content = content[docstring_match.end():].lstrip()
    
    # Check for comment license (# lines at the beginning)
    lines = content.split('\n')
    start_code = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith('#'):
            start_code = i
            break
    
    # If we have comments at the top and they contain license info
    if start_code > 0:
        comment_block = '\n'.join(lines[:start_code])
        if 'Copyright' in comment_block or 'License' in comment_block:
            content = '\n'.join(lines[start_code:])
    
    return content

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    
    cleaned_content = remove_license_header(content)
    
    with open(sys.argv[2], 'w') as f:
        f.write(cleaned_content)
EOF
        
        # Run the Python script to remove the license header
        python3 "$python_script" "$file" "$temp_file"
        rm "$python_script"
    else
        # No license header found, just copy the file
        cat "$file" > "$temp_file"
    fi
    
    # Create the final file with the new license header
    final_file="${file}.final"
    
    # Add shebang if it existed
    if [ $has_shebang -eq 1 ]; then
        echo "$shebang_line" > "$final_file"
        echo "" >> "$final_file"
    else
        # Start with empty file
        > "$final_file"
    fi
    
    # Add the license header
    echo "$LICENSE_HEADER" >> "$final_file"
    echo "" >> "$final_file"
    
    # Add the rest of the file content
    cat "$temp_file" >> "$final_file"
    
    # Replace the original file
    mv "$final_file" "$file"
    rm -f "$temp_file"
    
    echo "  Done"
done

echo "License update complete!"
