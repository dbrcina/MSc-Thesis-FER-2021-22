from typing import List

import torch


def ctc_decoder(log_probs: torch.Tensor, blank: int = 0) -> List[int]:
    labels = log_probs.argmax(dim=-1)

    reconstructed = []
    previous_label = None
    for label in labels:
        if label != blank:
            if label != previous_label:
                reconstructed.append(label.item())
                previous_label = label

    return reconstructed
