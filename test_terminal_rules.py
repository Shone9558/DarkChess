"""Headless rule, search, GUI termination and self-play integration tests."""
import ast
import contextlib
import copy
import io
from pathlib import Path
import random
import types
import unittest
from unittest.mock import patch

import numpy as np

from config import CONFIG
from game import Board, Game, move_action2move_id
from mcts import MCTS, MCTSPlayer, TreeNode
from mcts_pure import MCTS as PureMCTS


def action(text):
    return move_action2move_id[text]


def position(pieces=None, limit=15, first=False, remaining=()):
    b = Board()
    b.init_board(start_player=1)
    grid = [['一一'] * 8 for _ in range(4)]
    for (y, x), piece in (pieces or {(0, 0): '红车', (3, 7): '黑车'}).items():
        grid[y][x] = piece
    for i in range(len(b.state_deque)):
        b.state_deque[i] = copy.deepcopy(grid)
    b.state_list = copy.deepcopy(grid)
    b.remain_pieces = list(remaining)
    b.first_move = first
    b.no_progress_limit = limit
    b.reset_terminal_history()
    return b


CYCLE = ['0001', '3736', '0100', '3637']


class ScriptBoard(Board):
    def __init__(self, initial):
        self.initial = initial

    def init_board(self, start_player=None):
        state = copy.deepcopy(self.initial.__dict__)
        self.__dict__.update(state)


class ScriptPlayer:
    def __init__(self, moves):
        self.moves = moves
        self.resets = 0

    def set_player_ind(self, player):
        self.player = player

    def reset_player(self):
        self.resets += 1

    def get_action(self, b, temp=1, return_prob=False):
        if b.game_end()[0]:
            raise AssertionError('Controller requested a move after game end')
        move = action(self.moves[b.action_count])
        if return_prob:
            probs = np.zeros(384)
            probs[move] = 1
            return move, probs
        return move


class TerminalRulesTests(unittest.TestCase):
    def test_no_legal_moves_loses_even_with_more_material(self):
        b = position({(0, 0): '红帅', (0, 1): '黑兵', (1, 0): '黑兵'})
        self.assertGreater(b.calc_side_strength('红'), b.calc_side_strength('黑'))
        self.assertEqual(b.game_end(), (True, 2))
        self.assertEqual(b.terminal_reason, 'no_legal_moves')
        self.assertEqual(b.has_a_winner(), b.game_end())

    def test_black_with_no_pieces_loses(self):
        b = position({(0, 0): '红车'})
        b.current_player_id, b.current_player_color = 2, '黑'
        b.reset_terminal_history()
        self.assertEqual(b.game_end(), (True, 1))

    def test_covered_square_is_a_legal_action(self):
        b = position({(0, 0): '暗棋', (3, 7): '黑车'}, remaining=['红兵'])
        self.assertEqual(b.game_end(), (False, -1))
        self.assertIn(action('0000'), b.availables)

    def test_third_occurrence_draws_and_reverse_moves_are_legal(self):
        b = position()
        for i, text in enumerate(CYCLE * 2):
            self.assertIn(action(text), b.availables)
            b.do_move(action(text))
            self.assertEqual(b.game_end()[0], i == 7)
        self.assertEqual(b.game_end(), (True, -1))
        self.assertEqual(b.terminal_reason, 'threefold_repetition')
        self.assertEqual(b._position_counts[b.position_key()], 3)

    def test_key_includes_side_hidden_counts_and_flip_eligibility(self):
        b = position(remaining=['红兵', '黑帅'])
        original = b.position_key()
        b.remain_pieces.reverse()
        self.assertEqual(original, b.position_key())
        b.current_player_id, b.current_player_color = 2, '黑'
        self.assertNotEqual(original, b.position_key())
        b.current_player_id, b.current_player_color = 1, '红'
        b.remain_pieces.append('红兵')
        self.assertNotEqual(original, b.position_key())
        b.remain_pieces.pop()
        b.first_move = True
        self.assertNotEqual(original, b.position_key())

    def test_no_progress_draws_at_exact_ply_limit_not_material_score(self):
        b = position({(0, 0): '红车', (3, 7): '黑兵'}, limit=3)
        for i, text in enumerate(['0001', '3736', '0102'], 1):
            b.do_move(action(text))
            self.assertEqual(b.kill_action, i)
            self.assertEqual(b.game_end()[0], i == 3)
        self.assertEqual(b.game_end(), (True, -1))
        self.assertEqual(b.terminal_reason, 'no_progress_limit')

    def test_capture_and_reveal_reset_clock(self):
        b = position({(0, 0): '红车', (0, 1): '黑兵', (3, 7): '黑车'})
        b.kill_action = 14
        self.assertGreater(b.do_move(action('0001')), 0)
        self.assertEqual(b.kill_action, 0)
        b = position({(0, 0): '暗棋', (3, 7): '黑车'}, first=True, remaining=['红兵'])
        b.kill_action = 14
        b.do_move(action('0000'), revealed_piece='红兵')
        self.assertEqual(b.kill_action, 0)
        self.assertFalse(b.game_end()[0])

    def test_precedence_and_sticky_result(self):
        b = position({(0, 0): '黑车'}, limit=1)
        b.kill_action = 1
        b._position_counts[b.position_key()] = 3
        self.assertEqual(b.game_end(), (True, 2))
        self.assertEqual(b.terminal_reason, 'no_legal_moves')
        self.assertEqual(b.game_end(), (True, 2))
        b = position(limit=8)
        for text in CYCLE * 2:
            b.do_move(action(text))
        self.assertEqual(b.terminal_reason, 'threefold_repetition')

    def test_invalid_and_post_terminal_moves_do_not_change_history(self):
        b = position()
        before = copy.deepcopy(b.__dict__)
        with self.assertRaises(ValueError):
            b.do_move(action('0010'), revealed_piece='红兵')
        self.assertEqual(b.__dict__, before)
        with self.assertRaises(ValueError):
            b.do_move(action('0002'))
        self.assertEqual(b.__dict__, before)
        b.no_progress_limit = 1
        b.do_move(action('0001'))
        before = copy.deepcopy(b.__dict__)
        with self.assertRaises(ValueError):
            b.do_move(action('3736'))
        self.assertEqual(b.__dict__, before)

    def test_new_game_resets_history_and_freezes_config(self):
        b = position(limit=1)
        b.do_move(action('0001'))
        with patch.dict(CONFIG, kill_action=9, repetition_limit=3):
            b.init_board(start_player=2)
        self.assertEqual(b.no_progress_limit, 9)
        self.assertEqual(b.kill_action, 0)
        self.assertIsNone(b.winner)
        self.assertIsNone(b.terminal_reason)
        self.assertEqual(sum(b._position_counts.values()), 1)

    def test_search_copies_do_not_pollute_real_history(self):
        b = position()
        branch = copy.deepcopy(b)
        for text in CYCLE * 2:
            branch.do_move(action(text))
        self.assertTrue(branch.game_end()[0])
        self.assertFalse(b.game_end()[0])
        self.assertEqual(sum(b._position_counts.values()), 1)

    def test_mcts_checks_terminal_before_policy_and_existing_children(self):
        for draw in [True, False]:
            b = position(limit=1) if draw else position({(0, 0): '黑车'})
            if draw:
                b.do_move(action('0001'))
            policy = unittest.mock.Mock(side_effect=AssertionError('Terminal policy evaluated'))
            search = MCTS(policy, n_playout=2)
            search._root._children[123] = TreeNode(search._root, 1)
            search._playout(b)
            self.assertEqual(search._root._Q, 0 if draw else 1)
            self.assertEqual(search.get_move_probs(b)[0], ())
            with self.assertRaises(ValueError):
                MCTSPlayer(policy).get_action(b)
            pure = PureMCTS(policy)
            pure._playout(b)
            self.assertEqual(pure._root._Q, 0 if draw else 1)
            policy.assert_not_called()

    def test_search_stops_when_a_move_reaches_draw(self):
        b = position(limit=1)
        policy = unittest.mock.Mock(side_effect=AssertionError('Terminal policy evaluated'))
        search = MCTS(policy)
        search._root.expand([(action('0001'), 1)], b)
        search._playout(b)
        self.assertEqual(search._root._children[action('0001')]._Q, 0)
        self.assertEqual(b.game_end(), (True, -1))

    def test_selfplay_and_match_share_results_and_labels(self):
        cases = [
            (position(), CYCLE * 2, -1, 'threefold_repetition'),
            (position({(0, 0): '红车', (0, 1): '黑兵', (3, 7): '黑车'}, limit=2),
             ['0001', '3736', '0102'], -1, 'no_progress_limit'),
            (position({(0, 0): '红车', (0, 1): '黑兵'}),
             ['0001'], 1, 'no_legal_moves'),
            (position({(0, 0): '红兵', (0, 2): '黑车'}),
             ['0001', '0201'], 2, 'no_legal_moves'),
        ]
        for initial, moves, expected, reason in cases:
            with self.subTest(reason=reason), contextlib.redirect_stdout(io.StringIO()):
                b = ScriptBoard(initial)
                player = ScriptPlayer(moves)
                winner, data = Game(b).start_self_play(player)
                samples = list(data)
                self.assertEqual(winner, expected)
                self.assertEqual(b.terminal_reason, reason)
                self.assertEqual(len(samples), len(moves))
                self.assertEqual([z for _, _, z in samples],
                                 [0 if expected == -1 else (1 if expected == i % 2 + 1 else -1)
                                  for i in range(len(moves))])
                self.assertEqual(player.resets, 1)
                b = ScriptBoard(initial)
                self.assertEqual(Game(b).start_play(ScriptPlayer(moves), ScriptPlayer(moves), is_shown=0), expected)
                self.assertEqual(b.terminal_reason, reason)

    def test_already_terminal_controllers_do_not_request_moves(self):
        initial = position({(0, 0): '黑车'})
        with contextlib.redirect_stdout(io.StringIO()):
            player = ScriptPlayer([])
            self.assertEqual(Game(ScriptBoard(initial)).start_play(player, player, is_shown=0), 2)
            winner, data = Game(ScriptBoard(initial)).start_self_play(player)
            self.assertEqual(winner, 2)
            self.assertEqual(list(data), [])

    def test_actual_gui_termination_block_uses_same_result(self):
        # Run the unmodified UI's actual result block without opening Pygame.
        tree = ast.parse(Path(__file__).with_name('UIplay.py').read_text(encoding='utf-8'))
        loop = next(n for n in tree.body if isinstance(n, ast.While))
        start = next(i for i, n in enumerate(loop.body)
                     if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'game_end')
        code = compile(ast.Module(body=loop.body[start:], type_ignores=[]), 'GUI terminal block', 'exec')
        def quit_gui():
            raise SystemExit
        repeated = position()
        for text in CYCLE * 2:
            repeated.do_move(action(text))
        limited = position(limit=1)
        limited.do_move(action('0001'))
        for b, expected in [(position({(0, 0): '黑车'}), 2), (limited, -1), (repeated, -1)]:
            scope = {'board': b, 'players': {1: 'red', 2: 'black'}, 'sys': types.SimpleNamespace(exit=quit_gui)}
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                exec(code, scope)
            self.assertEqual(scope['winner'], expected)

    def test_minimax_restores_terminal_history_and_obeys_result(self):
        # Load the actual class without unrelated GUI/API/neural dependencies.
        tree = ast.parse(Path(__file__).with_name('players.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MinimaxDarkChessPlayer')
        scope = {'copy': copy, 'random': random, 'Game': Game, 'move_action2move_id': move_action2move_id}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'Minimax class', 'exec'), scope)
        player = scope['MinimaxDarkChessPlayer']()
        b = position(limit=1)
        before = copy.deepcopy(b.__dict__)
        backup = player._backup_board(b)
        b.do_move(action('0001'))
        self.assertEqual(player.search_upward(b, 2, -99999, 99999, 2, 2), 0)
        player._restore_board(b, backup)
        self.assertEqual(b.__dict__, before)
        b = position({(0, 0): '暗棋', (3, 7): '黑车'}, first=True, remaining=['红兵'])
        before = copy.deepcopy(b.__dict__)
        player._simulate_reveal(b, (0, 0))
        self.assertEqual(b.__dict__, before)


if __name__ == '__main__':
    unittest.main()
