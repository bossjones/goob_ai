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

# Sort the array uniquely
declare -a sorted_unique_array=($(printf "%s\n" "${my_array[@]}" | sort -u))

# Print the array elements (for debugging purposes)
echo "Parsed function names:"
for func in "${sorted_unique_array[@]}"; do
    echo "func: ${func}"
done

echo -e "\n\n\n"


# Prompt the user for input
read -p "Do you want to proceed? (yes/no): " response

# Convert the response to lowercase
response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

# Check the user's response
if [[ "$response" == "yes" ]]; then
    echo "You chose yes. Proceeding..."
    # Add the commands you want to execute if the user says yes
elif [[ "$response" == "no" ]]; then
    echo "You chose no. Exiting..."
    exit 0
else
    echo "Invalid input. Please enter yes or no."
fi

set -e
# Iterate over the array and run the aider command to annotate first
for func in "${sorted_unique_array[@]}"; do
    echo "cmd: aider --message \"add typing annotations to function ${func} in ${ONLY_FILE_NAME}. Be sure to include return types when necessary.\" ${PYTHON_FILE}"
    aider --message "add typing annotations to function ${func} in ${ONLY_FILE_NAME}. Be sure to include return types when necessary." "${PYTHON_FILE}"
done

echo -e "\n\n\n"

# Iterate over the array and run the aider command
for func in "${sorted_unique_array[@]}"; do
    # echo "func: ${func}"
    echo "cmd: aider --message \"add descriptive docstrings to function ${func} in ${ONLY_FILE_NAME}. Please use pep257 convention. Update existing docstrings if need be.\" ${PYTHON_FILE}"
    aider --message "add descriptive docstrings to function ${func} in ${ONLY_FILE_NAME}. Please use pep257 convention. Update existing docstrings if need be." "${PYTHON_FILE}"
done

echo -e "\n\n\n"
