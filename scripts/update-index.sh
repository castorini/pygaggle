#!/bin/bash

echo "Updating Anserini index..."

INDEX_NAME=${1:-lucene-index-cord19-paragraph-2020-05-12}
INDEX_URL=${2:-https://www.dropbox.com/s/z8s0urul6l4zig2/lucene-index-cord19-paragraph-2020-05-12.tar.gz}

wget ${INDEX_URL}
tar xvfz ${INDEX_NAME}.tar.gz && rm ${INDEX_NAME}.tar.gz

INDEX_PATH=indexes/${INDEX_NAME}
rm -rf ${INDEX_PATH}
mv ${INDEX_NAME} ${INDEX_PATH}

echo "Successfully updated Anserini index at indexes/${INDEX_NAME}"
