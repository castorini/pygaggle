from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ClosedBookQA:

    def __init__(self, model: str = "google/t5-large-ssm", device = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device).eval()

    def predict(self, question: str):
        encode = self.tokenizer(question, return_tensors='pt', padding=True).to(self.device)
        answer = self.model.generate(**encode)
        decoded = self.tokenizer.batch_decode(answer, skip_special_tokens=True)
        return decoded[0]
