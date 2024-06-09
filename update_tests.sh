#!/usr/bin/env bash
# USAGE: ./update_tests.sh /path/to/your/directory
# USAGE: ./update_tests.sh ./src/goob_ai/utils/imgops.py tests/utils/test_imgops.py

# TODO: Get this working, almost there
# for i in $(fd . -t d -d 5 './src/goob_ai'); do
#     # new_path=$(python3 -c "import os;p=os.path.relpath('${i}',start='./src');print(p)")
#     new_path=$(python3 -c "import os;p=os.path.join('tests', os.path.relpath('${i}',start='./src/goob_ai'));print(p)")
#     echo "Rendering ${new_path}"
#     echo "mkdir -p \"${new_path}\" || true"
#     mkdir -p "${new_path}" || true
#     touch "${new_path}/__init__.py" || true
# done

# SRC_DIR="./src/goob_ai"
# TEST_DIR="./tests"

# # Function to create the test file
# create_test_file() {
#     local src_file="$1"
#     local relative_path="${src_file#$SRC_DIR/}"
#     local test_file="$TEST_DIR/${relative_path%.py}.py"
#     local test_file_formatted=$(echo $test_file | sed 's/\/\(.*\)\.py$/\/test_\1.py/')

#     echo -e "\n\n\n"
#     echo "src_file: $src_file"
#     echo "relative_path: $relative_path"
#     echo "test_file: $test_file"
#     echo "test_file_formatted: $test_file_formatted"
#     echo -e "\n\n\n"

#     # Create the directory structure if it doesn't exist
#     # mkdir -p "$(dirname "$test_file")" || true
#     echo "dirname: $(dirname "$test_file")"
#     echo "mkdir -p \"$(dirname \"$test_file\")\""
#     # Touch the test file
#     # touch "$test_file"
#     echo "touch \"$test_file\""
#     echo "Created test file: $test_file"
# }

# for i in $(fd . -t f -E '__init__.py' -e py './src/goob_ai'); do
#     # File path to be tested
#     src_file="${i}"
#     # Create the corresponding test file
#     create_test_file "$src_file"
# done

###############################################################################3


# -d, --max-depth <depth>
# for i in $(fd . --type f --max-depth 5 './src'); do
#     new_path=$(python3 -c "import os;p=os.path.relpath('${i}');print(p)")
#     echo "Rendering ${new_path}"
#     # mkdir -p "${new_path}" || true
#     echo "mkdir -p \"${new_path}\" || true"
# done

# for j in $(fd . -H -t f -e j2 -d 5 './src'); do
#     old_path=$(python3 -c "import os;p=os.path.join(os.path.relpath('${j}'));print(p)")
#     new_path=$(python3 -c "import os;p=os.path.join('outputs', os.path.relpath('${j}',start='./src')).replace('.j2','');print(p)")

#     echo "Rendering ${old_path} -> ${new_path}"
#     echo "jinja render -d ./data.yaml -t \"./${old_path}\" -o \"./${new_path}\""
#     jinja render -d ./data.yaml -t "./${old_path}" -o "./${new_path}"
# done


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
    echo "cmd: aider --message \"using pytest, write tests for ${TEST_ONLY_FILE_NAME} that cover function ${func} in ${ONLY_FILE_NAME}. mock if necessary but only use pytest-mock, do not use unittest.mock. Use pytest-asyncio where necessary as well. If it involves python modules Pillow or cv2 use fixture tests/fixtures/screenshot_image_larger00013.PNG as your test image.\" ${PYTHON_FILE} ${TEST_PYTHON_FILE}"
    # aider --message "using pytest, write tests for ${TEST_ONLY_FILE_NAME} that cover function ${func} in ${ONLY_FILE_NAME}. mock if necessary but only use pytest-mock, do not use unittest.mock. Use pytest-asyncio where necessary as well. If it involves python modules Pillow or cv2 use fixture tests/fixtures/screenshot_image_larger00013.PNG as your test image." "${PYTHON_FILE}" "${TEST_PYTHON_FILE}"
    aider --message "using pytest,  update tests in ${TEST_ONLY_FILE_NAME} that cover function ${func} for module ${ONLY_FILE_NAME} to not use mocker any functions involving python modules cv2 and Pillow" "${PYTHON_FILE}" "${TEST_PYTHON_FILE}"
done

echo -e "\n\n\n"
