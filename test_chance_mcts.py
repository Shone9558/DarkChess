import copy
import contextlib
import io
import random
import unittest
from unittest.mock import patch

import numpy as np

from config import CONFIG
from game import Board, Game, move_action2move_id
from mcts import MCTS, MCTSPlayer, TreeNode, ChanceNode
from mcts_pure import MCTS as PureMCTS, MCTS_Pure


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

    def test_same_legal_action_keeps_outcome_specific_priors_and_values(self):
        state = board()
        state.first_move = False
        common_action = move_action2move_id['0101']

        def outcome_policy(s):
            red = s.state_deque[-1][0][0] == '红兵'
            return [(a, 0.8 if red else 0.2) for a in s.availables], (0.7 if red else -0.4)

        search = MCTS(outcome_policy)
        search._root.expand([(FLIP, 1)], state)
        chance = search._root._children[FLIP]
        for piece in ['红兵', '黑帅']:
            with patch('game.random.choices', return_value=[piece]):
                search._playout(copy.deepcopy(state))
        red, black = chance._children['红兵'], chance._children['黑帅']
        self.assertIsNot(red._children[common_action], black._children[common_action])
        self.assertAlmostEqual(red._children[common_action]._P, 0.8)
        self.assertAlmostEqual(black._children[common_action]._P, 0.2)
        self.assertAlmostEqual(red._Q, -0.7)
        self.assertAlmostEqual(black._Q, 0.4)

    def test_chance_sampling_matches_counts_and_terminal_expectation(self):
        state = board()
        state.first_move = False
        state.remain_pieces = ['红兵'] * 3 + ['黑帅']
        search = MCTS(policy)
        search._root.expand([(FLIP, 1)], state)

        def terminal_after_reveal(s):
            piece = s.state_deque[-1][0][0]
            return (False, -1) if piece == '暗棋' else (True, 1 if piece == '红兵' else 2)

        random.seed(1234)
        with patch.object(Board, 'game_end', terminal_after_reveal):
            for _ in range(400):
                search._playout(copy.deepcopy(state))
        chance = search._root._children[FLIP]
        red, black = chance._children['红兵'], chance._children['黑帅']
        self.assertAlmostEqual(red._n_visits / 400, 0.75, delta=0.08)
        self.assertEqual(red._Q, 1)
        self.assertEqual(black._Q, -1)
        self.assertAlmostEqual(chance._Q, (red._n_visits - black._n_visits) / 400)
        self.assertAlmostEqual(chance._Q, 0.5, delta=0.16)
        self.assertEqual(chance._n_visits, red._n_visits + black._n_visits)

    def test_actual_board_terminal_stops_before_policy_or_existing_children(self):
        state = board()
        state.winner = 2
        for search_type in [MCTS, PureMCTS]:
            net = unittest.mock.Mock(side_effect=AssertionError('Terminal policy called'))
            search = search_type(net, n_playout=2)
            search._root.expand([(FLIP, 1)], state)
            before = copy.deepcopy(state.__dict__)
            search._playout(state)
            self.assertEqual(search._root._Q, 1)
            self.assertEqual(search.get_move_probs(state)[0], ())
            self.assertEqual(state.__dict__, before)
            net.assert_not_called()
        with self.assertRaises(ValueError):
            MCTSPlayer(policy).get_action(state)

    def test_output_excludes_stale_illegal_children(self):
        state = board()
        search = MCTS(policy, n_playout=1)
        search.get_move_probs(state)
        illegal = move_action2move_id['0001']
        search._root._children[illegal] = TreeNode(search._root, 1)
        search._root._children[illegal]._n_visits = 100000
        acts, probs = search.get_move_probs(state)
        self.assertNotIn(illegal, acts)
        self.assertEqual(set(acts), set(state.availables))
        self.assertAlmostEqual(float(sum(probs)), 1)

    def test_player_waits_for_actual_reveal_then_reuses_matching_outcome(self):
        state = board()
        player = MCTSPlayer(policy, n_playout=1, is_selfplay=1)
        root = player.mcts._root
        root.expand([(FLIP, 1)], state)
        chance = root._children[FLIP]
        with patch('game.random.choices', return_value=['红兵']):
            player.mcts._playout(copy.deepcopy(state))
        expected = chance._children['红兵']
        with patch.object(player.mcts, 'get_move_probs', return_value=((FLIP,), np.array([1.0]))):
            move = player.get_action(state)
        self.assertIs(player.mcts._root, root)
        state.do_move(move, revealed_piece='红兵')
        player.observe_move(state, move)
        self.assertIs(player.mcts._root, expected)
        self.assertIsNone(player.mcts._root._parent)
        self.assertIsNone(player._pending_move)

    def test_missing_observation_resets_stale_root_before_next_search(self):
        state = board()
        player = MCTSPlayer(policy, n_playout=2, is_selfplay=1)
        move = player.get_action(state)
        old_root = player.mcts._root
        state.do_move(move)
        next_move = player.get_action(state)
        self.assertIsNot(player.mcts._root, old_root)
        self.assertIn(next_move, state.availables)

    def test_invalid_reveals_and_normal_move_outcome_are_atomic(self):
        state = board()
        with self.assertRaises(ValueError):
            state.get_reveal_outcomes(move_action2move_id['0001'])
        state.do_move(FLIP, revealed_piece='红兵')
        state.do_move(move_action2move_id['0101'], revealed_piece='黑帅')
        before = copy.deepcopy(state.__dict__)
        for action, piece in [(FLIP, '红兵'), (move_action2move_id['0001'], '红兵')]:
            with self.assertRaises(ValueError):
                state.do_move(action, revealed_piece=piece)
            self.assertEqual(state.__dict__, before)

    def test_sampling_uses_counts_not_hidden_order_for_both_modes(self):
        original = board()
        original.remain_pieces = ['黑帅', '红兵', '红兵', '红帅']
        for simulate in [False, True]:
            state = copy.deepcopy(original)
            reversed_state = copy.deepcopy(original)
            reversed_state.remain_pieces.reverse()
            self.assertEqual(dict(state.get_reveal_outcomes(FLIP)),
                             dict(reversed_state.get_reveal_outcomes(FLIP)))
            with patch('game.random.choices', return_value=['红兵']) as sample:
                state.do_move(FLIP, is_simulate=simulate)
            self.assertEqual(sample.call_args.kwargs['weights'], (2 / 3, 1 / 3))
            self.assertEqual(state.remain_pieces.count('红兵'), 1)

    def test_pure_search_splits_outcomes_and_keeps_real_board_unchanged(self):
        state = board()
        state.first_move = False
        before = copy.deepcopy(state.__dict__)
        search = PureMCTS(policy, n_playout=3)
        search._root.expand([(FLIP, 1)], state)
        with patch.object(search, '_evaluate_rollout', return_value=0.25):
            for piece in ['红兵', '黑帅']:
                with patch('game.random.choices', return_value=[piece]):
                    search._playout(copy.deepcopy(state))
        chance = search._root._children[FLIP]
        self.assertIsInstance(chance, ChanceNode)
        self.assertEqual(set(chance._children), {'红兵', '黑帅'})
        self.assertEqual(chance._n_visits, 2)
        self.assertEqual(state.__dict__, before)
        # Also run real rollouts, rather than only the mocked value path.
        random.seed(91)
        np.random.seed(91)
        action = PureMCTS(policy, n_playout=3).get_move(state)
        self.assertIn(action, state.availables)
        self.assertEqual(state.__dict__, before)

    def test_full_open_board_has_only_decision_children(self):
        state = Board()
        with patch.dict(CONFIG, no_dark_mode=True):
            state.init_board(start_player=1)
        search = MCTS(policy, n_playout=3)
        acts, _ = search.get_move_probs(state)
        self.assertTrue(acts)
        self.assertTrue(all(not isinstance(node, ChanceNode)
                            for node in search._root._children.values()))

    def test_temperature_schedule_and_reset_survive_delayed_reuse(self):
        state = board()
        player = MCTSPlayer(policy, n_playout=1, is_selfplay=1)
        player.playout_count = 29
        with patch.object(player.mcts, 'get_move_probs', return_value=((FLIP,), np.array([1.0]))) as get:
            player.get_action(state)
            self.assertEqual(get.call_args.args[1], 1.0)
            player.get_action(state)
            self.assertEqual(get.call_args.args[1], 0.001)
            player.reset_player()
            self.assertIsNone(player._pending_move)
            player.get_action(state)
            self.assertEqual(get.call_args.args[1], 1.0)
        other = MCTSPlayer(policy, n_playout=1)
        with patch.object(other.mcts, 'get_move_probs', return_value=((FLIP,), np.array([1.0]))) as get:
            other.get_action(state, temp=0.25)
            self.assertEqual(get.call_args.args[1], 0.25)
        self.assertTrue(other.mcts._root.is_leaf())

    def test_selfplay_controller_reports_every_actual_move(self):
        initial = board()
        grid = [['一一'] * 8 for _ in range(4)]
        grid[0][0], grid[0][1], grid[3][7] = '暗棋', '暗棋', '黑兵'
        for i in range(len(initial.state_deque)):
            initial.state_deque[i] = copy.deepcopy(grid)
        initial.state_list = copy.deepcopy(grid)
        initial.remain_pieces = ['红车', '黑帅']

        class FixedBoard(Board):
            def init_board(self, start_player=None):
                self.__dict__ = copy.deepcopy(initial.__dict__)

        class CheckingPlayer(MCTSPlayer):
            def __init__(self):
                super().__init__(policy, n_playout=8, is_selfplay=1)
                self.observations = 0

            def observe_move(self, b, move):
                self.observations += 1
                assert b.action_count == self.observations
                assert b.last_move == move
                super().observe_move(b, move)

            def get_action(self, b, **kwargs):
                assert self._pending_move is None, 'Previous result was not reported'
                assert b.action_count < 100, 'Unexpectedly unbounded test game'
                move, probs = super().get_action(b, **kwargs)
                assert move in b.availables
                assert set(np.flatnonzero(probs)).issubset(set(b.availables))
                return move, probs

        random.seed(32)
        np.random.seed(32)
        player = CheckingPlayer()
        with patch.dict(CONFIG, kill_action=3), contextlib.redirect_stdout(io.StringIO()):
            winner, data = Game(FixedBoard()).start_self_play(player)
        samples = list(data)
        self.assertIn(winner, [-1, 1, 2])
        self.assertGreater(len(samples), 1)
        self.assertEqual(player.observations, len(samples))
        self.assertEqual(player.playout_count, 0)
        self.assertIsNone(player._pending_move)
        self.assertTrue(player.mcts._root.is_leaf())


if __name__ == '__main__':
    unittest.main()
