#!/usr/bin/env bash
# USAGE: ./update_tests.sh /path/to/your/directory
# USAGE: ./update_tests.sh ./src/goob_ai/utils/imgops.py tests/utils/test_imgops.py

# Check if directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_python_file> <path_to_test_python_file>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Usage: $0 <path_to_python_file>  <path_to_test_python_file>"
    exit 1
fi

# eg. src/goob_ai/goob_bot.py
PATH_TO_PYTHON_FILE=$1
# eg. src/goob_ai/goob_bot.py
PYTHON_FILE="${PATH_TO_PYTHON_FILE}"
# eg. goob_bot.py
ONLY_FILE_NAME=$(basename "${PYTHON_FILE}")


# eg. src/goob_ai/goob_bot.py
TEST_PATH_TO_PYTHON_FILE=$2
# eg. src/goob_ai/goob_bot.py
TEST_PYTHON_FILE="${TEST_PATH_TO_PYTHON_FILE}"
# eg. goob_bot.py
TEST_ONLY_FILE_NAME=$(basename "${TEST_PYTHON_FILE}")

touch "${TEST_PYTHON_FILE}" || true


# Check if the Python file exists
if [ ! -f "$PYTHON_FILE" ]; then
    echo "File $PYTHON_FILE not found!"
    exit 1
fi
# Check if the Python file exists
if [ ! -f "$TEST_PYTHON_FILE" ]; then
    echo "File $TEST_PYTHON_FILE not found!"
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

# echo "using pytest, write tests for test_file_functions.py that cover every function in file_functions.py. mock if necessary but only use pytest-mock, do not use unittest.mock. Use pytest-asyncio where necessa
# echo "using pytest, write tests for ${TEST_ONLY_FILE_NAME} that cover function ${func} in ${ONLY_FILE_NAME}. mock if necessary but only use pytest-mock, do not use unittest.mock. Use pytest-asyncio where necessary as well. If it involves python modules Pillow or cv2 use fixture tests/fixtures/screenshot_image_larger00013.PNG as your test image."

# Iterate over the array and run the aider command to annotate first
for func in "${sorted_unique_array[@]}"; do
    echo "cmd: aider --message \"using pytest, write integration tests for ${TEST_ONLY_FILE_NAME} that cover function ${func} in ${ONLY_FILE_NAME}. Make sure you are using @pytest.mark.integration to indicate that it is an integration test. mock if necessary but only use pytest-mock, do not use unittest.mock. Use pytest-asyncio if and only if ${func} is an async function. If it involves python modules Pillow or cv2 use fixture tests/fixtures/screenshot_image_larger00013.PNG as your test image. If the test involves pdfs use src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf\" ${PYTHON_FILE} ${TEST_PYTHON_FILE}"


    aider --message "using pytest, write tests for ${TEST_ONLY_FILE_NAME} that cover function ${func} in ${ONLY_FILE_NAME}. mock if necessary but only use pytest-mock, do not use unittest.mock. Use pytest-asyncio if and only if ${func} is an async function. If it involves python modules Pillow or cv2 use fixture tests/fixtures/screenshot_image_larger00013.PNG as your test image. If the test involves pdfs use src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf" ${PYTHON_FILE} ${TEST_PYTHON_FILE}
done

echo -e "\n\n\n"
