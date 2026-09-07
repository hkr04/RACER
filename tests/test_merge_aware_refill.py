import unittest

from racer.automaton import Automaton


VOCAB_SIZE = 1024
TOP_K = 4


def k_ary_children(token, k=TOP_K):
    return [k * token + j for j in range(1, k + 1)]


def prefix_set(candidates):
    prefixes = set()
    for path in candidates:
        for i in range(1, len(path) + 1):
            prefixes.add(tuple(path[:i]))
    return prefixes


def update_rows(ac, mapping, top_k=TOP_K):
    tokens = []
    vectors = []
    for token, children in mapping.items():
        row = list(children) + [0] * top_k
        tokens.append(token)
        vectors.append(row[:top_k])
    ac.update(tokens, vectors)


def logits_only_automaton(vocab_size=VOCAB_SIZE, top_k=TOP_K):
    ac = Automaton(1)
    ac.init_logits(vocab_size, top_k)
    tokens = list(range(vocab_size))
    adj_vectors = [k_ary_children(token, top_k) for token in tokens]
    ac.update(tokens, adj_vectors)
    return ac


def assert_tree_attention(test_case, buf):
    for indices in buf.retrieve_indices:
        ancestors = []
        for depth, idx in enumerate(indices):
            ancestors.append(idx)
            test_case.assertEqual(buf.position_ids[idx], depth)
            visible = {
                j for j, value in enumerate(buf.attn_mask[idx])
                if value == 1
            }
            test_case.assertEqual(visible, set(ancestors))


class TestMergeAwareRefill(unittest.TestCase):
    def test_logits_root_already_present_does_not_consume_budget(self):
        # Retrieval already inserts x. Old remaining = C - selected.size()
        # still charged x again inside TokenBin, leaving a merge hole.
        x = 10
        capacity = 4
        ac = Automaton(32)
        ac.init_logits(64, TOP_K)
        update_rows(ac, {
            x: [20, 21, 22, 23],
            20: [30, 31, 32, 33],
            21: [34, 35, 36, 37],
            22: [38, 39, 40, 41],
            23: [42, 43, 44, 45],
        })
        ac.insert([1, x], freq=3)
        ac.build()
        ac.trans_tokens([1])

        buf = ac.retrieve(x, capacity)

        self.assertEqual(len(buf.tree_candidates), capacity)
        self.assertIn((x,), prefix_set(buf.candidates))

    def test_existing_prefix_continues_downward(self):
        x, a, b = 10, 20, 30
        ac = Automaton(32)
        ac.init_logits(64, TOP_K)
        update_rows(ac, {
            x: [a, 21, 22, 23],
            a: [b, 31, 32, 33],
            b: [40, 41, 42, 43],
        })
        ac.insert([1, x, a], freq=5)
        ac.build()
        ac.trans_tokens([1])

        buf = ac.retrieve(x, 8)

        self.assertIn((x, a, b), prefix_set(buf.candidates))
        self.assertEqual(len(buf.tree_candidates), 8)

    def test_duplicate_logits_tokens_do_not_repeat_budget_or_enqueue(self):
        x, a, b, c, d, e = 10, 20, 30, 31, 32, 33
        capacity = 6
        ac = Automaton(1)
        ac.init_logits(64, TOP_K)
        update_rows(ac, {
            x: [a, a, a, a],
            a: [b, c, d, e],
        })

        buf = ac.retrieve(x, capacity)

        prefixes = prefix_set(buf.candidates)
        self.assertEqual(len(buf.tree_candidates), capacity)
        self.assertEqual({path for path in prefixes if len(path) == 2}, {(x, a)})
        self.assertEqual(
            {path for path in prefixes if len(path) == 3},
            {(x, a, b), (x, a, c), (x, a, d), (x, a, e)},
        )

    def test_capacity_is_filled_when_logits_can_expand(self):
        ac = logits_only_automaton(vocab_size=65536)
        for capacity in (1, 2, 8, 21, 64):
            buf = ac.retrieve(0, capacity)
            self.assertEqual(len(buf.tree_candidates), capacity)
            self.assertEqual(len(prefix_set(buf.candidates)), capacity)

    def test_underfilled_when_no_valid_extension(self):
        ac = Automaton(8)
        ac.init_logits(16, TOP_K)
        buf = ac.retrieve(99, 8)

        self.assertEqual(len(buf.tree_candidates), 1)
        self.assertEqual(set(map(tuple, buf.candidates)), {(99,)})

    def test_tree_attention_after_refill(self):
        ac = logits_only_automaton()
        buf = ac.retrieve(0, 21)
        assert_tree_attention(self, buf)

        x, a, b = 10, 20, 30
        ac = Automaton(32)
        ac.init_logits(64, TOP_K)
        update_rows(ac, {
            x: [a, 21, 22, 23],
            a: [b, 31, 32, 33],
        })
        ac.insert([1, x, a], freq=5)
        ac.build()
        ac.trans_tokens([1])
        buf = ac.retrieve(x, 8)
        assert_tree_attention(self, buf)

    def test_retrieval_merge_hole_is_given_to_logits_refill(self):
        # Two borders share the same next_token root, so selected states
        # collapse to one unique node. Refill should consume the hole.
        x = 10
        capacity = 4
        ac = Automaton(64)
        ac.init_logits(64, TOP_K)
        update_rows(ac, {
            x: [20, 21, 22, 23],
            20: [30, 31, 32, 33],
            21: [34, 35, 36, 37],
            22: [38, 39, 40, 41],
            23: [42, 43, 44, 45],
        })
        ac.insert([1, 2, x], freq=5)
        ac.insert([2, x], freq=5)
        ac.build()
        ac.trans_tokens([1, 2])

        buf = ac.retrieve(x, capacity)

        self.assertEqual(len(buf.tree_candidates), capacity)
        self.assertIn((x,), prefix_set(buf.candidates))

    def test_chain_mode_still_fills_capacity(self):
        ac = logits_only_automaton(vocab_size=65536)
        buf = ac.retrieve(0, 8, is_chain=True)

        self.assertEqual(len(buf.tree_candidates), 8)
        self.assertEqual(len(buf.candidates), 1)
        self.assertEqual(len(buf.candidates[0]), 8)
        assert_tree_attention(self, buf)

    def test_chain_mode_hybrid_keeps_retrieval_prefix(self):
        x, a, b = 10, 20, 30
        ac = Automaton(32)
        ac.init_logits(64, TOP_K)
        update_rows(ac, {
            x: [a, 21, 22, 23],
            a: [b, 31, 32, 33],
            b: [40, 41, 42, 43],
        })
        ac.insert([1, x, a], freq=5)
        ac.build()
        ac.trans_tokens([1])

        buf = ac.retrieve(x, 8, is_chain=True)

        self.assertLessEqual(len(buf.tree_candidates), 8)
        self.assertIn((x, a), prefix_set(buf.candidates))
        self.assertTrue(
            any(tuple(path[:2]) == (x, a) for path in buf.candidates),
            buf.candidates,
        )
        assert_tree_attention(self, buf)


if __name__ == "__main__":
    unittest.main()
