This document describes how to convert our t5 to huggingface model and then upload to remote
### Load T5 Tensorflow model to transformers.
```python
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
# load t5 tf model to transformers model
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained("path/to/model.ckpt-1011000.index", from_tf=True, config=config)

# get t5 tokenizer ready too
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# save model & tokenizer together to the folder to upload to huggingface
model.save_pretrained("path/to/model/")
tokenizer.save_pretrained("path/to/model/")
```

### Using `transformers-cli` to upload the model folder to huggingface
1. Use `transformers-cli login` to login
2. Upload the model to huggingface under our organization
```
transformers-cli upload path/to/model/ --organization castorini
```

Please see [transformers](https://huggingface.co/transformers/model_sharing.html) for details.