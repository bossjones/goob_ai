#!/usr/bin/env bash
# shellcheck shell=bash
# SOURCE: https://github.com/microsoft/CyberBattleSim/blob/main/createstubs.sh
# write a description of the script here
# This script generates type stubs for the installed packages using pyright.
# See pyright documentation for more information: https://microsoft.github.io/pyright/#/command-line

set -e

pushd "$(dirname "$0")"

echo "$(tput setaf 2)Creating type stubs$(tput sgr0)"
createstub() {
    local name=$1
    if [ ! -d "typings/$name" ]; then
        pyright --createstub "$name"
    else
        echo stub "$name" already created
    fi
}
param=$1
if [[ $param == "--recreate" ]]; then
    echo 'Deleting typing directory'
    rm -Rf typings/
fi

echo 'Creating stubs'

mkdir -p typings/ || true

createstub pandas
createstub plotly
createstub progressbar
createstub pytest
createstub setuptools
createstub ordered_set
createstub asciichartpy
createstub networkx
createstub boolean
createstub IPython



# cat temp | sed -n 's/.*Stub file not found for "\([^"]*\)".*/\1/p'
rye run typecheck | grep "Stub file not found for" | sed -n 's/.*Stub file not found for "\([^"]*\)".*/\1/p' | sed -n '/\./ s/\([^.]*\)\..*/\1/p' | sort | uniq | xargs -I {} echo "pyright --createstub {}"

# pyright --createstub better_exceptions
# pyright --createstub bpdb
# pyright --createstub pinecone
# pyright --createstub discord
# pyright --createstub transformers
# pyright --createstub uritools
# pyright --createstub torchvision
# pyright --createstub scipy
# pyright --createstub webcolors
# pyright --createstub logging_tree

# if [ ! -d "typings/gym" ]; then
#     pyright --createstub gym
#     # Patch gym stubs
#     echo '    spaces = ...' >> typings/gym/spaces/dict.pyi
#     echo '    nvec = ...' >> typings/gym/spaces/space.pyi
#     echo '    spaces = ...' >> typings/gym/spaces/space.pyi
#     echo '    spaces = ...' >> typings/gym/spaces/tuple.pyi
#     echo '    n = ...' >> typings/gym/spaces/multi_binary.pyi
# else
#     echo stub gym already created
# fi


echo 'Typing stub generation completed'

popd
