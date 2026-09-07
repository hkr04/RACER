import unittest

from racer.automaton import Automaton


VOCAB_SIZE = 1024
TOP_K = 4


def k_ary_children(token, k=TOP_K):
    return [k * token + j for j in range(1, k + 1)]


def logits_adj_vectors(vocab_size=VOCAB_SIZE, top_k=TOP_K):
    tokens = list(range(vocab_size))
    adj_vectors = [k_ary_children(token, top_k) for token in tokens]
    return tokens, adj_vectors


def logits_only_automaton(vocab_size=VOCAB_SIZE, top_k=TOP_K):
    ac = Automaton(1)
    ac.init_logits(vocab_size, top_k)
    tokens, adj_vectors = logits_adj_vectors(vocab_size, top_k)
    ac.update(tokens, adj_vectors)
    return ac


FIGURE9_LEAF_PATHS = {
    (0, 4, 17),
    (0, 1, 5, 21),
    (0, 1, 5, 22),
    (0, 1, 6, 25),
    (0, 1, 7, 29),
    (0, 1, 8, 33),
    (0, 2, 9, 37),
    (0, 2, 10, 41),
    (0, 3, 13, 53),
}


class TestAutomatonPaperAlignment(unittest.TestCase):
    def test_figure9_algorithm1_eq3_4ary_tree(self):
        ac = logits_only_automaton()
        buf = ac.retrieve(0, 21)

        self.assertEqual(len(buf.tree_candidates), 21)
        self.assertEqual(set(tuple(path) for path in buf.candidates), FIGURE9_LEAF_PATHS)

    def test_tree_attention_eq2(self):
        ac = logits_only_automaton()
        buf = ac.retrieve(0, 21)

        for indices in buf.retrieve_indices:
            ancestors = []
            for depth, idx in enumerate(indices):
                ancestors.append(idx)
                self.assertEqual(buf.position_ids[idx], depth)

                visible = {
                    j for j, value in enumerate(buf.attn_mask[idx])
                    if value == 1
                }
                self.assertEqual(visible, set(ancestors))

    def test_paper_default_border_depth_is_2(self):
        ac = Automaton(32)
        ac.insert([1, 2, 3], freq=3)
        ac.build()

        buf = ac.retrieve(1, 8)
        self.assertEqual(set(map(tuple, buf.candidates)), {(1,)})

        ac = Automaton(32)
        ac.insert([1, 2, 3], freq=3)
        ac.build()
        ac.trans_tokens([1])

        buf = ac.retrieve(2, 8)
        self.assertIn((2, 3), set(map(tuple, buf.candidates)))

    def test_max_nodes_zero_constructor_does_not_crash(self):
        ac = Automaton(0)
        buf = ac.retrieve(7, 1)
        self.assertEqual(set(map(tuple, buf.candidates)), {(7,)})

    def test_draft_capacity_is_not_exceeded(self):
        # Vocab must be large enough that k-ary children stay in-range
        # through capacity 64; otherwise expansion stops at unseen rows.
        ac = logits_only_automaton(vocab_size=65536)
        for capacity in (1, 2, 21, 64):
            buf = ac.retrieve(0, capacity)
            self.assertLessEqual(len(buf.tree_candidates), capacity)
            self.assertEqual(len(buf.tree_candidates), capacity)


if __name__ == "__main__":
    unittest.main()
