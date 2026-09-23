import unittest

import torch
import torch.nn.functional as F

from racer.model.utils import evaluate_posterior, top_p_filtering


def nucleus_probs(logits_row, temperature, top_p):
    row = (logits_row.unsqueeze(0) / temperature).clone()
    return F.softmax(top_p_filtering(row, top_p=top_p), dim=-1)[0]


def emitted_token(logits, candidates, temperature, top_p, pad_token_id):
    best, length, next_token = evaluate_posterior(
        logits, candidates, temperature, top_p, pad_token_id=pad_token_id
    )
    if length.item() >= 1:
        return int(candidates[best, 1]), next_token
    return int(next_token), next_token


class NucleusRejectionTest(unittest.TestCase):
    def test_greedy_has_no_residual_token(self):
        logits = torch.tensor([[[0.0, 2.0, 1.0], [0.0, 0.0, 3.0]]])
        candidates = torch.tensor([[0, 1]])
        _, length, next_token = evaluate_posterior(logits, candidates, temperature=0, top_p=0)
        self.assertIsNone(next_token)
        self.assertEqual(length.item(), 1)

    def test_certain_draft_is_always_accepted(self):
        logits = torch.full((1, 2, 4), -1e4)
        logits[0, 0, 2] = 10.0
        candidates = torch.tensor([[0, 2, 9]])
        for _ in range(20):
            best, length, next_token = evaluate_posterior(
                logits, candidates, temperature=1.0, top_p=0.9, pad_token_id=9
            )
            self.assertEqual(length.item(), 1)
            self.assertEqual(int(candidates[best, 1]), 2)
            self.assertIsNone(next_token)

    def test_marginal_matches_nucleus(self):
        temperature = 1.0
        top_p = 0.999
        pad = 9
        # Position 0 is the distribution under test. Position 1 is unused filler.
        base = torch.log(torch.tensor([0.1, 0.4, 0.35, 0.15]))
        logits = base.view(1, 1, 4).repeat(2, 2, 1).clone()
        # Two siblings, then a padding slot. Order is the acceptance order.
        candidates = torch.tensor(
            [
                [0, 1, pad],
                [0, 2, pad],
            ]
        )
        target = nucleus_probs(base, temperature, top_p)
        counts = torch.zeros(4)
        n = 8000
        for _ in range(n):
            token, next_token = emitted_token(logits, candidates, temperature, top_p, pad)
            counts[token] += 1
            if next_token is not None:
                self.assertNotIn(int(next_token), (1, 2))
        freq = counts / n
        self.assertTrue(torch.allclose(freq, target, atol=0.03), f"{freq} vs {target}")

    def test_chain_does_not_bias_the_first_token(self):
        temperature = 0.7
        top_p = 0.9
        probs = torch.tensor([0.05, 0.25, 0.5, 0.2])
        base = torch.log(probs)
        logits = base.view(1, 1, 4).repeat(1, 3, 1).clone()
        candidates = torch.tensor([[0, 2, 1]])
        target = nucleus_probs(base, temperature, top_p)
        counts = torch.zeros(4)
        n = 8000
        for _ in range(n):
            token, _ = emitted_token(logits, candidates, temperature, top_p, pad_token_id=None)
            counts[token] += 1
        freq = counts / n
        self.assertTrue(torch.allclose(freq, target, atol=0.03), f"{freq} vs {target}")


if __name__ == "__main__":
    unittest.main()
