from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ClosedBookQA:

    def __init__(self, pretrained_model_name_or_path: str = "google/t5-large-ssm"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)

    def predict(self, question: str):
        encode = self.tokenizer(question, return_tensors='pt', padding=True)
        answer = self.model.generate(**encode)
        decoded = self.tokenizer.batch_decode(answer, skip_special_tokens=True)
        return decoded[0]
