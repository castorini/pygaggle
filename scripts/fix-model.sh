#!/bin/bash

mv bert_config.json config.json
for filename in model.ckpt*; do
    mv $filename $(python -c "import re; print(re.sub(r'ckpt-\\d+', 'ckpt', '$filename'))");
done