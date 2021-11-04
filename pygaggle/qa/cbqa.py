from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ClosedBookQA:

    def __init__(self, model: str = "google/t5-large-ssm", device = 'cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device).eval()

    def predict(self, question: str):
        encode = self.tokenizer(question, return_tensors='pt').to(self.device)
        answer = self.model.generate(encode.input_ids)[0]
        decoded = self.tokenizer.decode(answer, skip_special_tokens=True)
        return decoded
