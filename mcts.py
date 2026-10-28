"""暗棋 MCTS：決策節點選動作，機會節點按機率抽取翻棋結果。"""

import copy

import numpy as np

from config import CONFIG
from game import get_all_legal_moves_darkchess

move_id2move_action, move_action2move_id = get_all_legal_moves_darkchess()


def softmax(x):
    probs = np.exp(x - np.max(x))
    return probs / np.sum(probs)


class TreeNode:
    """已知盤面的決策節點；子節點按動作編號索引。

    非根節點的 Q 是「抵達此盤面的上一手玩家」的價值，供父層選動作。
    翻棋結果節點與其父 ChanceNode 使用同一位翻棋玩家的視角。
    根節點的 Q 不參與選擇，只使用其訪問次數計算探索項。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0.0
        self._u = 0.0
        self._P = prior_p

    def expand(self, action_priors, state):
        """一般動作接決策節點；翻棋動作先接機會節點。"""
        for action, prob in action_priors:
            if action not in self._children:
                node_type = ChanceNode if state.is_reveal_move(action) else TreeNode
                self._children[action] = node_type(self, prob)

    def select(self, c_puct, legal_moves):
        legal_moves = set(legal_moves)
        valid_children = [(a, n) for a, n in self._children.items() if a in legal_moves]
        if not valid_children:
            return None, None
        return max(valid_children, key=lambda item: item[1].get_value(c_puct))

    def get_value(self, c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits)
                   / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """只在跨越玩家回合時反號，翻棋結果到機會節點不反號。"""
        if self._parent is not None:
            parent_value = leaf_value if isinstance(self._parent, ChanceNode) else -leaf_value
            self._parent.update_recursive(parent_value)
        self.update(leaf_value)

    def is_leaf(self):
        return not self._children

    def is_root(self):
        return self._parent is None


class ChanceNode(TreeNode):
    """一個翻棋動作；子節點以棋種索引，不能用 PUCT 挑選結果。

    本節點的 N/Q 彙整所有抽樣結果；各結果的後續搜尋統計彼此獨立。
    結果按真實機率抽樣，因此 Q 直接取樣本平均，不再乘一次機率。
    """

    def sample(self, state, action):
        probabilities = dict(state.get_reveal_outcomes(action))
        # Board 統一負責抽樣、扣除棋子、歷史盤面與換手，避免兩套翻棋規則。
        state.do_move(action, is_simulate=True)
        y, x, _, _ = map(int, move_id2move_action[action])
        piece = state.state_deque[-1][y][x]
        if piece not in self._children:
            self._children[piece] = TreeNode(self, probabilities[piece])
        return self._children[piece]

    def select(self, c_puct, legal_moves):
        raise TypeError('Chance outcomes must be sampled, not selected with PUCT')

    def expand(self, action_priors, state):
        raise TypeError('Chance outcomes are created by sample()')


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _evaluate_leaf(self, state, network_value):
        """回傳葉節點當前玩家的價值；純 MCTS 可改用 rollout。"""
        return float(np.asarray(network_value).item())

    def _playout(self, state):
        node = self._root
        while True:
            # 必須先檢查終局，不能走入既有子節點或再呼叫網路。
            end, winner = state.game_end()
            if end:
                leaf_value = (0.0 if winner == -1 else
                              1.0 if winner == state.get_current_player_id() else -1.0)
                break

            if node.is_leaf():
                action_probs, network_value = self._policy(state)
                node.expand(action_probs, state)
                leaf_value = self._evaluate_leaf(state, network_value)
                break

            action, next_node = node.select(self._c_puct, state.availables)
            if action is None:
                # 防禦性處理：不沿用與盤面不符的子節點。
                node._children.clear()
                continue
            if isinstance(next_node, ChanceNode):
                node = next_node.sample(state, action)
            else:
                state.do_move(action, is_simulate=True)
                node = next_node

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3, plot_probs=True):
        """翻棋結果在機會節點抽樣；根部輸出仍是原有的動作機率。"""
        if temp <= 0 or not np.isfinite(temp):
            raise ValueError('temp must be finite and positive')
        if self._n_playout < 1:
            raise ValueError('n_playout must be at least 1')
        if state.game_end()[0]:
            return (), np.array([], dtype=float)
        legal_moves = set(state.availables)
        if not legal_moves:
            return (), np.array([], dtype=float)

        for _ in range(self._n_playout):
            # 不再預先洗牌決定整串結果，每次翻棋都依當時的剩餘棋種抽樣。
            self._playout(copy.deepcopy(state))

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()
                      if act in legal_moves]
        if not act_visits:
            return (), np.array([], dtype=float)
        acts, visits = zip(*act_visits)
        act_probs = softmax(np.log(np.array(visits, dtype=float) + 1e-10) / temp)
        return acts, act_probs

    def update_with_move(self, last_move, revealed_piece=None):
        """只沿用實際動作及已觀察到的翻棋結果；未知結果從新根開始。"""
        next_root = self._root._children.get(last_move)
        if isinstance(next_root, ChanceNode):
            next_root = next_root._children.get(revealed_piece)
        if next_root is None:
            self._root = TreeNode(None, 1.0)
        else:
            self._root = next_root
            self._root._parent = None

    def __str__(self):
        return 'MCTS'


class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=3, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = 'AI'
        self.playout_count = 0
        self._pending_move = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)
        self.playout_count = 0
        self._pending_move = None

    def observe_move(self, board, move):
        """由自我對弈控制器在 Board.do_move 後通知實際結果。"""
        if not self._is_selfplay:
            return
        if (self._pending_move != (move, board.action_count - 1)
                or board.last_move != move):
            self.mcts.update_with_move(-1)
        else:
            y, x, to_y, to_x = map(int, move_id2move_action[move])
            piece = board.state_deque[-1][y][x] if (y, x) == (to_y, to_x) else None
            self.mcts.update_with_move(move, revealed_piece=piece)
        self._pending_move = None

    def __str__(self):
        return 'MCTS {}'.format(self.player)

    def get_action(self, board, temp=1e-3, return_prob=0):
        if board.game_end()[0] or not board.availables:
            raise ValueError('Cannot select a move from a terminal board or without legal moves')
        if self._pending_move is not None:
            # 舊呼叫端若未通知結果，就重新搜尋，不能誤用翻棋前的根節點。
            self.mcts.update_with_move(-1)
            self._pending_move = None
        move_probs = np.zeros(384)

        current_temp = temp
        if self._is_selfplay:
            self.playout_count += 1
            current_temp = 1.0 if self.playout_count <= 30 else 1e-3

        acts, probs = self.mcts.get_move_probs(board, current_temp)
        if not acts:
            raise ValueError('Search returned no legal moves')
        eat_ids = []
        for a in acts:
            y1, x1, y2, x2 = map(int, move_id2move_action[a])
            target = board.state_deque[-1][y2][x2]
            if target not in ('一一', '暗棋') and board.current_player_color not in target:
                eat_ids.append(a)
        if eat_ids:
            # 保留原本的吃子偏好，本次只修正翻棋搜尋結構。
            eat_index = [i for i, a in enumerate(acts) if a in eat_ids]
            probs = probs.copy()
            probs[eat_index] *= 40
            probs /= probs.sum()
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            move = np.random.choice(
                acts,
                p=0.8 * probs + 0.2 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            ) if current_temp == 1.0 else np.random.choice(acts, p=probs)
            # 此時尚未翻棋，等控制器回報結果後才移動根節點。
            self._pending_move = (move, board.action_count)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        return move
