from typing import Union, Tuple

from transformers import PreTrainedModel
import torch

__all__ = ['greedy_decode']

DecodedOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def greedy_decode(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  length: int,
                  attention_mask: torch.Tensor = None,
                  return_last_logits: bool = True) -> DecodedOutput:
    decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True)
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat([decode_ids,
                                next_token_logits.max(1)[1].unsqueeze(-1)],
                               dim=-1)
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids
