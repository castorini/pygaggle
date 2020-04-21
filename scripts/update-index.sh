#!/bin/bash

echo "Updating Anserini index..."

INDEX_NAME=lucene-index-covid-paragraph-2020-04-10
INDEX_URL=https://www.dropbox.com/s/ivk87journyajw3/lucene-index-covid-paragraph-2020-04-10.tar.gz

wget ${INDEX_URL}
tar xvfz ${INDEX_NAME}.tar.gz && rm ${INDEX_NAME}.tar.gz

rm -rf data/lucene-index-covid-paragraph
mv ${INDEX_NAME} data/lucene-index-covid-paragraph

echo "Successfully updated Anserini index at data/${INDEX_NAME}"
