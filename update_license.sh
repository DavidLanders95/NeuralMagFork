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
    
    # Remove any existing license header
    # This pattern looks for common license indicators at the top of the file
    if grep -q -E "^(#|\"\"\"|\'\'\').*([Cc]opyright|[Ll]icense)" "$file" | head -20; then
        echo "  Removing existing license header..."
        
        # For docstring-style headers (triple quotes)
        if head -20 "$file" | grep -q -E "^(\"\"\"|''').*[Cc]opyright"; then
            # Skip the docstring license block
            awk 'BEGIN{skip=0; found=0}
                /^"""/ || /^'\'''\''/ {
                    if (skip == 0 && !found && ($0 ~ /[Cc]opyright/ || NR < 20)) {
                        skip=1;
                        found=1;
                        next;
                    }
                    if (skip == 1) {
                        skip=0;
                        next;
                    }
                }
                skip == 0 {print}' "$file" > "$temp_file"
        else
            # For comment-style headers
            awk 'BEGIN{skip=1}
                skip==1 && /^[^#]/ {skip=0}
                skip==1 && /^$/ {skip=0}
                skip==0 || !/^#/ {print}' "$file" > "$temp_file"
        fi
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
    if [ $has_shebang -eq 1 ]; then
        tail -n +2 "$temp_file" >> "$final_file"
    else
        cat "$temp_file" >> "$final_file"
    fi
    
    # Replace the original file
    mv "$final_file" "$file"
    rm -f "$temp_file"
    
    echo "  Done"
done

echo "License update complete!"
