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

# Create a license docstring for Python files
LICENSE_DOCSTRING='"""MIT License

Copyright (c) 2024 NeuralMag team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""'

# Find all Python files in the neuralmag directory
find ./neuralmag -type f -name "*.py" \
    -not -path "*/__pycache__*" \
    | while read file; do
    
    echo "Processing $file"
    
    # Create a temporary file
    temp_file="${file}.tmp"
    
    # Check for existing license header patterns
    if grep -q "Copyright" "$file"; then
        echo "  License already exists, updating..."
        
        # Check if it's a docstring license (triple quotes)
        if grep -q '""".*Copyright' "$file" || grep -q "'''.*Copyright" "$file"; then
            # Extract content after the first triple-quoted section
            awk 'BEGIN{docstring=0; printed=0} 
                /^"""/ || /^'\'''\''/ {
                    if (docstring == 0) {
                        docstring=1
                    } else if (docstring == 1) {
                        docstring=2
                        next
                    }
                }
                docstring == 2 && !printed {print ""; printed=1; next}
                docstring == 2 {print}' "$file" > "$temp_file"
            
            # Add license docstring at the beginning
            final_temp="${file}.final"
            echo "$LICENSE_DOCSTRING" > "$final_temp"
            echo "" >> "$final_temp"
            cat "$temp_file" >> "$final_temp"
            mv "$final_temp" "$file"
            rm -f "$temp_file"
        else
            # It's a comment-style license
            # Find where the license ends (first blank line or non-comment line)
            awk 'BEGIN{in_license=1}
                /^[^#]/ && in_license {in_license=0}
                /^$/ && in_license {in_license=0}
                !in_license {print}' "$file" > "$temp_file"
            
            # Add license header at the beginning
            final_temp="${file}.final"
            echo "$LICENSE_HEADER" > "$final_temp"
            echo "" >> "$final_temp"
            cat "$temp_file" >> "$final_temp"
            mv "$final_temp" "$file"
            rm -f "$temp_file"
        fi
    else
        echo "  Adding new license header"
        
        # Check if the file starts with a shebang or coding declaration
        if head -1 "$file" | grep -q "^#!"; then
            # Extract the shebang line
            head -1 "$file" > "$temp_file"
            echo "" >> "$temp_file"
            echo "$LICENSE_HEADER" >> "$temp_file"
            echo "" >> "$temp_file"
            tail -n +2 "$file" >> "$temp_file"
            mv "$temp_file" "$file"
        else
            # No shebang, just add the license header
            echo "$LICENSE_HEADER" > "$temp_file"
            echo "" >> "$temp_file"
            cat "$file" >> "$temp_file"
            mv "$temp_file" "$file"
        fi
    fi
    
    echo "  Done"
done

echo "License update complete!"
