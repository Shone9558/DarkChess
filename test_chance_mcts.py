import copy
import random
import unittest
from unittest.mock import patch

import numpy as np

from config import CONFIG
from game import Board, move_action2move_id
from mcts import MCTS, TreeNode, ChanceNode


FLIP = move_action2move_id['0000']


def board():
    result = Board()
    with patch.dict(CONFIG, no_dark_mode=False):
        result.init_board(start_player=1)
    return result


def policy(state):
    actions = state.availables
    return [(a, 1 / len(actions)) for a in actions], 0.25


class ChanceMCTSTests(unittest.TestCase):
    def test_probabilities_and_first_flip_rule(self):
        state = board()
        state.remain_pieces = ['红兵', '红兵', '红帅', '黑帅']
        self.assertEqual(dict(state.get_reveal_outcomes(FLIP)),
                         {'红兵': 2 / 3, '红帅': 1 / 3})
        state.first_move = False
        self.assertEqual(dict(state.get_reveal_outcomes(FLIP)),
                         {'红兵': 0.5, '红帅': 0.25, '黑帅': 0.25})
        with patch('game.random.choices', return_value=['黑帅']) as sample:
            state.do_move(FLIP)
        self.assertEqual(sample.call_args.kwargs['weights'], (0.5, 0.25, 0.25))

    def test_explicit_reveal_updates_complete_state(self):
        state = board()
        previous = copy.deepcopy(state.state_deque[-1])
        count = state.remain_pieces.count('红兵')
        self.assertEqual(state.do_move(FLIP, revealed_piece='红兵'), 0)
        self.assertEqual(state.state_deque[-1][0][0], '红兵')
        self.assertEqual(state.state_deque[-2], previous)
        self.assertEqual(state.remain_pieces.count('红兵'), count - 1)
        self.assertEqual(state.current_player_id, 2)
        self.assertEqual(state.action_count, 1)
        self.assertEqual(state.last_move, FLIP)
        self.assertFalse(state.first_move)

    def test_invalid_outcome_does_not_mutate_state(self):
        state = board()
        before = copy.deepcopy(state.__dict__)
        with self.assertRaises(ValueError):
            state.do_move(FLIP, revealed_piece='黑帅')
        self.assertEqual(state.__dict__, before)
        state.remain_pieces = ['黑帅']
        with self.assertRaises(ValueError):
            state.do_move(FLIP)

    def test_distinct_outcomes_keep_distinct_legal_continuations(self):
        state = board()
        state.first_move = False
        # Leave room for a revealed black piece to move on black's turn.
        for position in state.state_deque:
            position[0][1] = '一一'
        state.remain_pieces.remove('红兵')
        search = MCTS(policy)
        search._root.expand([(FLIP, 1)], state)
        chance = search._root._children[FLIP]
        self.assertIsInstance(chance, ChanceNode)
        for piece in ['红兵', '黑帅']:
            with patch('game.random.choices', return_value=[piece]):
                search._playout(copy.deepcopy(state))
            expected = copy.deepcopy(state)
            expected.do_move(FLIP, revealed_piece=piece)
            self.assertEqual(set(chance._children[piece]._children), set(expected.availables))
        red, black = [chance._children[p] for p in ['红兵', '黑帅']]
        self.assertIsNot(red, black)
        self.assertNotEqual(set(red._children), set(black._children))
        black_visits = black._n_visits
        with patch('game.random.choices', return_value=['红兵']):
            search._playout(copy.deepcopy(state))
        self.assertEqual(black._n_visits, black_visits)
        self.assertEqual(red._n_visits, 2)
        self.assertEqual(chance._n_visits, 3)
        self.assertEqual(state.state_deque[-1][0][0], '暗棋')

    def test_backup_signs_and_action_aggregation(self):
        root = TreeNode(None, 1)
        chance = ChanceNode(root, 1)
        first, second = TreeNode(chance, 0.5), TreeNode(chance, 0.5)
        first.update_recursive(0.8)
        second.update_recursive(-0.2)
        self.assertAlmostEqual(chance._Q, 0.3)
        self.assertAlmostEqual(root._Q, -0.3)
        continuation = TreeNode(first, 1)
        continuation.update_recursive(0.4)
        self.assertAlmostEqual(first._Q, 0.2)
        self.assertEqual(chance._n_visits, 3)
        self.assertEqual(root._n_visits, 3)

    def test_tree_reuse_requires_known_outcome(self):
        for piece in [None, '红兵', '黑帅']:
            search = MCTS(policy)
            search._root.expand([(FLIP, 1)], board())
            chance = search._root._children[FLIP]
            outcome = TreeNode(chance, 1)
            chance._children['红兵'] = outcome
            search.update_with_move(FLIP, revealed_piece=piece)
            self.assertIsNone(search._root._parent)
            if piece == '红兵':
                self.assertIs(search._root, outcome)
            else:
                self.assertIsNot(search._root, outcome)
                self.assertTrue(search._root.is_leaf())

    def test_normal_move_tree_reuse(self):
        state = board()
        state.do_move(FLIP, revealed_piece='红兵')
        state.do_move(move_action2move_id['0101'], revealed_piece='黑帅')
        action = move_action2move_id['0001']
        search = MCTS(policy)
        search._root.expand([(action, 1)], state)
        child = search._root._children[action]
        self.assertNotIsInstance(child, ChanceNode)
        search.update_with_move(action)
        self.assertIs(search._root, child)

    def test_search_smoke_and_input_unchanged(self):
        random.seed(73)
        state = board()
        before = copy.deepcopy(state.__dict__)
        search = MCTS(policy, n_playout=150)
        actions, probabilities = search.get_move_probs(state, temp=1)
        self.assertEqual(set(actions), set(state.availables))
        self.assertAlmostEqual(float(np.sum(probabilities)), 1)
        self.assertEqual(state.__dict__, before)
        self.assertEqual(sum(n._n_visits for n in search._root._children.values()), 149)


if __name__ == '__main__':
    unittest.main()
