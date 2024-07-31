#!/usr/bin/env bash
# USAGE: ./update_ruff.sh <path_to_changelog_file> <path_to_pyproject_file> <base_version>
# USAGE: ./update_ruff.sh https://raw.githubusercontent.com/astral-sh/ruff/main/CHANGELOG.md ./pyproject.toml 0.4.8

# Check if directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_python_file> <path_to_test_python_file>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Usage: $0 <path_to_python_file>  <path_to_test_python_file>"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Usage: $0 <path_to_python_file>  <path_to_test_python_file>"
    exit 1
fi

# eg. https://raw.githubusercontent.com/astral-sh/ruff/main/CHANGELOG.md
PATH_TO_CHANGELOG_FILE=$1
# eg. https://raw.githubusercontent.com/astral-sh/ruff/main/CHANGELOG.md
CHANGELOG_FILE="${PATH_TO_CHANGELOG_FILE}"
# eg. goob_bot.py
# ONLY_FILE_NAME=$(basename "${CHANGELOG_FILE}")
CHANGELOG_FILE_NAME="RUFF_CHANGELOG.md"


# eg. https://raw.githubusercontent.com/astral-sh/ruff/main/CHANGELOG.md
PATH_TO_PYPROJECT=$2
# eg. https://raw.githubusercontent.com/astral-sh/ruff/main/CHANGELOG.md
PYPROJECT_FILE="${PATH_TO_PYPROJECT}"
# eg. goob_bot.py
# TEST_ONLY_FILE_NAME=$(basename "${PYPROJECT_FILE}")

BASE_VERSION=$3

curl -L ${PATH_TO_CHANGELOG_FILE} > ${CHANGELOG_FILE_NAME}

# Check if the Python file exists
if [ ! -f "$CHANGELOG_FILE_NAME" ]; then
    echo "File $CHANGELOG_FILE_NAME not found!"
    exit 1
fi
# Check if the Python file exists
if [ ! -f "$PYPROJECT_FILE" ]; then
    echo "File $PYPROJECT_FILE not found!"
    exit 1
fi


echo "cmd: aider --message \"using ${CHANGELOG_FILE_NAME} analyze the changelog notes and determine if any changes need to be made to ${PYPROJECT_FILE} to upgrade safely from ${BASE_VERSION} to the latest version which is at the top of the ${CHANGELOG_FILE_NAME} file. Please ensure that any values are correctly updated" ${CHANGELOG_FILE_NAME} ${PYPROJECT_FILE}"

aider --message "using ${CHANGELOG_FILE_NAME} analyze the changelog notes and determine if any changes need to be made to ${PYPROJECT_FILE} to upgrade safely from ${BASE_VERSION} to the latest version which is at the top of the ${CHANGELOG_FILE_NAME} file. Please ensure that any values are correctly updated" ${CHANGELOG_FILE_NAME} ${PYPROJECT_FILE}"

echo -e "\n\n\n"
