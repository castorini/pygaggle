import abc

import torch

from pygaggle.model.encode import SingleEncoderOutput


__all__ = ['SimilarityMatrixProvider', 'CosineSimilarityMatrixProvider']


class SimilarityMatrixProvider:
    @abc.abstractmethod
    def compute_matrix(self,
                       encoded_query: SingleEncoderOutput,
                       encoded_document: SingleEncoderOutput) -> torch.Tensor:
        pass


class CosineSimilarityMatrixProvider(SimilarityMatrixProvider):
    @torch.no_grad()
    def compute_matrix(self,
                       encoded_query: SingleEncoderOutput,
                       encoded_document: SingleEncoderOutput) -> torch.Tensor:
        query_repr = encoded_query.encoder_output
        doc_repr = encoded_document.encoder_output
        matrix = torch.einsum('mh,nh->mn', query_repr, doc_repr)
        dnorm = doc_repr.norm(p=2, dim=1).unsqueeze(0)
        qnorm = query_repr.norm(p=2, dim=1).unsqueeze(1)
        matrix = (matrix / (qnorm + 1e-7)) / (dnorm + 1e-7)
        return matrix
