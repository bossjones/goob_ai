#!/usr/bin/env bash
# USAGE: ./update_docstrings.sh /path/to/your/directory
# USAGE: ./update_docstrings.sh ./src/goob_ai/goob_bot.py


# Check if directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_python_file>"
    exit 1
fi

# eg. src/goob_ai/goob_bot.py
PATH_TO_PYTHON_FILE=$1
# eg. src/goob_ai/goob_bot.py
PYTHON_FILE="${PATH_TO_PYTHON_FILE}"
# eg. goob_bot.py
ONLY_FILE_NAME=$(basename "${PYTHON_FILE}")

# Check if the Python file exists
if [ ! -f "$PYTHON_FILE" ]; then
    echo "File $PYTHON_FILE not found!"
    exit 1
fi

# # Parse out all function names and store them in an array
declare -a my_array=($(cat "$PYTHON_FILE" | grep -v "#" | ggrep -oP 'def \K\w+'))

# Parse out all function names that are not commented out and store them in an array
# declare -a my_array=($(ggrep -oP '^\s*def \K\w+' "$PYTHON_FILE"))
# declare -a my_array=($(cat "$PYTHON_FILE" | grep -v "#" | ggrep -oP '^\s*def \K\w+' "$PYTHON_FILE"))
# declare -a my_array=($(cat "$PYTHON_FILE" | grep -v "#" | ggrep -oP '^\s*def \K\w+'))


# Sort the array uniquely
declare -a sorted_unique_array=($(printf "%s\n" "${my_array[@]}" | sort -u))

# Print the array elements (for debugging purposes)
echo "Parsed function names:"
for func in "${sorted_unique_array[@]}"; do
    echo "$func"
done

echo -e "\n\n\n"

# Iterate over the array and run the aider command
for func in "${sorted_unique_array[@]}"; do
    # aider --message "add descriptive docstrings to function ${func} in ${ONLY_FILE_NAME}. Please use pep257 convention. Update existing docstrings if need be." ${PYTHON_FILE}
    echo "func: ${func}"
done
