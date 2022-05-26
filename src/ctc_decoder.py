from typing import List

import torch

from mappings import labels2text


def ctc_decoder(log_probs: torch.Tensor, mode: str = "greedy", beam_width: int = 5) -> List[str]:
    # batch_size, seq_length, classes
    log_probs = log_probs.permute(1, 0, 2)

    decoder = _greedy if mode == "greedy" else _beam_search

    decoded_list = []
    for log_prob in log_probs:
        decoded = decoder(log_prob, beam_width=beam_width)
        decoded_list.append(decoded)

    return decoded_list


def _beam_search(log_prob: torch.Tensor, beam_width: int) -> str:
    seq_length, classes_count = log_prob.shape

    beams = [([], 0)]

    for t in range(seq_length):
        all_candidates = []
        for beam_seq, beam_score in beams:
            for c in range(classes_count):
                class_log_prob = log_prob[t, c].item()
                if class_log_prob < torch.log(torch.tensor(0.1)):
                    continue
                candidate = [beam_seq + [c], beam_score + log_prob[t, c].item()]
                all_candidates.append(candidate)
        beams = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]

    return _reconstruct(beams[0][0])


def _greedy(log_prob: torch.Tensor, **kwargs) -> str:
    labels = log_prob.argmax(dim=-1)
    return _reconstruct(labels.view(-1).tolist())


def _reconstruct(labels: List[int], blank: int = 0) -> str:
    reconstructed = []
    previous_label = None
    for label in labels:
        if label != previous_label:
            reconstructed.append(label)
            previous_label = label

    reconstructed = list(filter(lambda x: x != blank, reconstructed))
    return labels2text(reconstructed)
