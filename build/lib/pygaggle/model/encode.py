from dataclasses import dataclass
from typing import List

from transformers import PreTrainedTokenizer
import torch
import torch.nn as nn

from .tokenize import BatchTokenizer
from pygaggle.rerank.base import TextType


__all__ = ['LongBatchEncoder', 'EncoderOutputBatch', 'SingleEncoderOutput',
           'SpecialTokensCleaner']


@dataclass
class SingleEncoderOutput:
    encoder_output: torch.Tensor
    token_ids: torch.Tensor
    text: TextType


@dataclass
class EncoderOutputBatch:
    encoder_output: List[torch.Tensor]
    token_ids: List[torch.Tensor]
    texts: List[TextType]

    def as_single(self) -> 'SingleEncoderOutput':
        return SingleEncoderOutput(self.encoder_output[0],
                                   self.token_ids[0], self.texts[0])

    def __iter__(self):
        return iter(SingleEncoderOutput(enc_out, token_ids, text) for
                    (enc_out, token_ids, text)
                    in zip(self.encoder_output, self.token_ids, self.texts))


class SpecialTokensCleaner:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.special_ids = tokenizer.all_special_ids

    def clean(self, output: SingleEncoderOutput) -> SingleEncoderOutput:
        indices = [idx for idx, tok in enumerate(output.token_ids.tolist())
                   if tok not in self.special_ids]
        return SingleEncoderOutput(output.encoder_output[indices],
                                   output.token_ids[indices], output.text)


class LongBatchEncoder:
    """
    Encodes batches of documents that are longer than the maximum sequence
    length by striding a window across
    the sequence dimension.
    Parameters
    ----------
    encoder : nn.Module
        The encoder module, such as `BertModel`.
    tokenizer : BatchTokenizer
        The batch tokenizer to use.
    max_seq_length : int
        The maximum sequence length, typically 512.
    """
    def __init__(self,
                 encoder: nn.Module,
                 tokenizer: BatchTokenizer,
                 max_seq_length: int = 512):
        self.encoder = encoder
        self.device = next(self.encoder.parameters()).device
        self.tokenizer = tokenizer
        self.msl = max_seq_length

    def encode_single(self, input: TextType) -> SingleEncoderOutput:
        return self.encode([input]).as_single()

    def encode(self, batch_input: List[TextType]) -> EncoderOutputBatch:
        batch_output = []
        batch_ids = []
        for ret in self.tokenizer.traverse(batch_input):
            input_ids = ret.output['input_ids']
            lengths = list(map(len, input_ids))
            batch_ids.extend(map(torch.tensor, input_ids))
            input_ids = [(idx, x) for idx, x in enumerate(input_ids)]
            max_len = min(max(lengths), self.msl)
            encode_lst = [[] for _ in input_ids]
            new_input_ids = [(idx, x[:max_len]) for idx, x in input_ids]
            while new_input_ids:
                attn_mask = [[1] * len(x[1]) +
                             [0] * (max_len - len(x[1]))
                             for x in new_input_ids]
                attn_mask = torch.tensor(attn_mask).to(self.device)
                nonpadded_input_ids = new_input_ids
                new_input_ids = [x + [0] * (max_len - len(x[:max_len]))
                                 for _, x in new_input_ids]
                new_input_ids = torch.tensor(new_input_ids).to(self.device)
                outputs, _ = self.encoder(input_ids=new_input_ids,
                                          attention_mask=attn_mask)
                for (idx, _), output in zip(nonpadded_input_ids, outputs):
                    encode_lst[idx].append(output)

                new_input_ids = [(idx, x[max_len:])
                                 for idx, x in nonpadded_input_ids
                                 if len(x) > max_len]
                max_len = min(max(map(lambda x: len(x[1]), new_input_ids),
                                  default=0), self.msl)

            encode_lst = list(map(torch.cat, encode_lst))
            batch_output.extend(encode_lst)
        return EncoderOutputBatch(batch_output, batch_ids, batch_input)
