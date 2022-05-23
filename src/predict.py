from typing import List

import torch
from torch.nn import functional as F

from ctc_decoder import ctc_decoder


def predict_binary(logits: torch.Tensor) -> torch.Tensor:
    assert logits.shape[-1] == 1
    return torch.sigmoid(logits)


def predict_ctc(logits: torch.Tensor) -> List[int]:
    log_probs = F.log_softmax(logits, dim=-1)
    return ctc_decoder(log_probs)
