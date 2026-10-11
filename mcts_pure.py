# -*- coding: utf-8 -*-
"""純 MCTS：共用暗棋機會節點搜尋，以隨機 rollout 取代神經網路價值。

Original pure MCTS implementation by Junxiao Song.
"""

from operator import itemgetter

import numpy as np

from mcts import MCTS as ChanceMCTS, TreeNode, ChanceNode


def rollout_policy_fn(board):
    actions = board.availables
    return zip(actions, np.random.rand(len(actions)))


def policy_value_fn(board):
    actions = board.availables
    if not actions:
        return [], 0.0
    return zip(actions, np.ones(len(actions)) / len(actions)), 0.0


class MCTS(ChanceMCTS):
    """選擇、結果分支和回傳與神經網路版一致，只替換葉節點估值。"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        super().__init__(policy_value_fn, c_puct, n_playout)

    def _evaluate_leaf(self, state, network_value):
        return self._evaluate_rollout(state)

    def _evaluate_rollout(self, state, limit=1000):
        player = state.get_current_player_id()
        for _ in range(limit):
            end, winner = state.game_end()
            if end:
                return 0.0 if winner == -1 else (1.0 if winner == player else -1.0)
            action_probs = list(rollout_policy_fn(state))
            if not action_probs:
                return -1.0 if state.get_current_player_id() == player else 1.0
            action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(action, is_simulate=True)
        # 最後一步也可能剛好終局；未結束的截斷 rollout 以中性值估計。
        end, winner = state.game_end()
        if end and winner != -1:
            return 1.0 if winner == player else -1.0
        return 0.0

    def get_move(self, state):
        acts, _ = self.get_move_probs(state)
        if not acts:
            raise ValueError('Cannot select a move from a terminal board or without legal moves')
        return max(acts, key=lambda action: self._root._children[action]._n_visits)


class MCTS_Pure:
    def __init__(self, c_puct=5, n_playout=500):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def agent(self):
        return 'AI'

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        move = self.mcts.get_move(board)
        self.mcts.update_with_move(-1)
        return move

    def __str__(self):
        return 'MCTS {}'.format(self.player)
