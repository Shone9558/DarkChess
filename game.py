"""棋盘游戏控制"""
import os

import numpy as np
import copy
import time

import torch
from matplotlib import pyplot as plt
from config import CONFIG
from collections import Counter, deque   # 这个队列用来判断长将或长捉
import random

from pytorch_net import PolicyValueNet


def generate_dark_chess_board():
    board = []
    for i in range(4):
        row = []
        for j in range(8):
            row.append('暗棋')
        board.append(row)

    return board

def load_latest_model(model_dir=".", use_checkpoint=True):
    """
    自動載入最新模型
    - 如果 use_checkpoint=True，優先找 checkpoint_xxx.pth
    - 否則使用 current_policy.pth
    """
    model_path = None
    # 如果沒有 checkpoint，或 use_checkpoint=False，就用 current_policy
    if model_path is None:
        model_path = os.path.join(model_dir, "current_policy.pth")

    print(f"載入模型: {model_path}")
    return PolicyValueNet(model_file=model_path)


def init_full_open_board():
    # 假設 all_pieces 是所有 32 個棋子的列表，例如 ["rP", "rP", ..., "bK"]
    red_pieces = ['红帅'] + ['红士'] * 2 + ['红象'] * 2 + ['红马'] * 2 + ['红车'] * 2 + ['红炮'] * 2 + ['红兵'] * 5
    black_pieces = ['黑帅'] + ['黑士'] * 2 + ['黑象'] * 2 + ['黑马'] * 2 + ['黑车'] * 2 + ['黑炮'] * 2 + ['黑兵'] * 5
    all_pieces = red_pieces + black_pieces
    random.shuffle(all_pieces)
    board = []
    index = 0
    for y in range(4):
        row = []
        for x in range(8):
            row.append(all_pieces[index])
            index += 1
        board.append(row)

    return board
    # 沒有暗棋了，所以 remain_pieces 清空
# 遊戲初始狀態
# 注意：在暗棋模式中，棋子位置是隨機生成的，這裡的 '红方在上' 僅為座標系參考
state_list_init = generate_dark_chess_board()
non_covered_state_list_init = init_full_open_board()

# deque来存储棋盘状态，长度为4
non_covered_state_deque_init = deque(maxlen=4)
state_deque_init = deque(maxlen=4)
for _ in range(4):
    state_deque_init.append(copy.deepcopy(state_list_init))
for _ in range(4):
    non_covered_state_deque_init.append(copy.deepcopy(non_covered_state_list_init))

# 构建一个字典：字符串到数组的映射，函数：数组到字符串的映射
string2array = dict(红车=np.array([1, 0, 0, 0, 0, 0, 0]), 红马=np.array([0, 1, 0, 0, 0, 0, 0]),
                    红象=np.array([0, 0, 1, 0, 0, 0, 0]), 红士=np.array([0, 0, 0, 1, 0, 0, 0]),
                    红帅=np.array([0, 0, 0, 0, 1, 0, 0]), 红炮=np.array([0, 0, 0, 0, 0, 1, 0]),
                    红兵=np.array([0, 0, 0, 0, 0, 0, 1]), 黑车=np.array([-1, 0, 0, 0, 0, 0, 0]),
                    黑马=np.array([0, -1, 0, 0, 0, 0, 0]), 黑象=np.array([0, 0, -1, 0, 0, 0, 0]),
                    黑士=np.array([0, 0, 0, -1, 0, 0, 0]), 黑帅=np.array([0, 0, 0, 0, -1, 0, 0]),
                    黑炮=np.array([0, 0, 0, 0, 0, -1, 0]), 黑兵=np.array([0, 0, 0, 0, 0, 0, -1]),
                    一一=np.array([0, 0, 0, 0, 0, 0, 0]), 暗棋=np.array([1, 1, 1, 1, 1, 1, 1]))


def array2string(array):
    return list(filter(lambda string: (string2array[string] == array).all(), string2array))[0]


# 改变棋盘状态
def change_state(state_list, move):
    """move : 字符串'0010'"""
    copy_list = copy.deepcopy(state_list)
    y, x, toy, tox = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    copy_list[toy][tox] = copy_list[y][x]
    copy_list[y][x] = '一一'
    return copy_list


# 打印盘面，可视化用到
def print_board(_state_array):
    # _state_array: [10, 9, 7], HWC
    board_line = []
    for i in range(4):
        for j in range(8):
            board_line.append(array2string(_state_array[i][j]))
        print(board_line)
        board_line.clear()


# 列表棋盘状态到数组棋盘状态
def state_list2state_array(state_list):
    _state_array = np.zeros([4, 8, 7])
    for i in range(4):
        for j in range(8):
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array


# 拿到所有合法走子的集合，384长度，也就是神经网络预测的走子概率向量的长度
# 第一个字典：move_id到move_action
# 第二个字典：move_action到move_id
# 例如：move_id:0 --> move_action:'0010'
def get_all_legal_moves_darkchess():
    _move_id2move_action = {}
    _move_action2move_id = {}
    idx = 0

    rows = 4
    cols = 8

    for r in range(rows):
        for c in range(cols):
            from_pos = f"{r}{c}"

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r, c
                while 0 <= nr < rows and 0 <= nc < cols:
                    to_pos = f"{nr}{nc}"
                    action = from_pos + to_pos
                    if action not in _move_action2move_id:
                        _move_id2move_action[idx] = action
                        _move_action2move_id[action] = idx
                        idx += 1
                    nr += dr
                    nc += dc


    return _move_id2move_action, _move_action2move_id

move_id2move_action, move_action2move_id = get_all_legal_moves_darkchess()


# 走子翻转的函数，用来扩充我们的数据
def flip_map(string):
    # string 是 4 位數字字串，例如 "0213"
    # 表示從 (2,1) -> (3,3)
    new_str = ''
    for index in range(4):
        digit = int(string[index])
        if index == 1 or index == 3:
            # 翻轉列（左右鏡像）
            digit = 7 - digit
        new_str += str(digit)
    return new_str

# 邊界檢查 (Legacy: 目前邏輯多已內含邊界判定)
def check_bounds(toY, toX):
    """檢查座標是否在 4x8 棋盤內"""
    if toY in [0, 1, 2, 3] and toX in [0, 1, 2, 3, 4, 5, 6, 7]:
        return True
    return False


# 不能走到自己的棋子位置
def check_obstruct(piece, current_player_color):
    # 当走到的位置存在棋子的时候，进行一次判断
    if piece != '一一':
        if current_player_color == '红':
            if '黑' in piece:
                return True
            else:
                return False
        elif current_player_color == '黑':
            if '红' in piece:
                return True
            else:
                return False
    else:
        return True


def get_legal_moves(state_deque, current_player_color):
    """
    計算當前盤面的所有合法移動。
    
    Args:
        state_deque: 包含歷史盤面狀態的隊列（用於判斷長打/長捉）。
        current_player_color: 當前玩家顏色 ('红' 或 '黑')。
        
    Returns:
        moves_id: 合法移動的 move_id 列表。
    """

    state_list = state_deque[-1]
    old_state_list = state_deque[-4]

    moves = []  # 用来存放所有合法的走子方法
    # state_list是以列表形式表示的, len(state_list) == 10, len(state_list[0]) == 9
    # 遍历移动初始位置
    rows, cols = 4, 8  # 暗棋盤面

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
    # 棋子等級 越大越強
    piece_strength = {
        '红炮': 2, '黑炮': 2,
        '红兵': 1, '黑兵': 1,
        '红马': 2, '黑马': 2,
        '红车': 3, '黑车': 3,
        '红象': 4, '黑象': 4,
        '红士': 5, '黑士': 5,
        '红帅': 6, '黑帅': 6,
        '暗棋': 10
    }
    # 棋子轉等級
    def get_strength(piece):
        for name in piece_strength:
            if name in piece:
                return piece_strength[name]
        return 0
    for i in range(rows):
        for j in range(cols):
            piece = state_list[i][j]
            if piece == '暗棋':
                m = str(i) + str(j) + str(i) + str(j)
                moves.append(m)
            elif piece != '一一' and current_player_color in piece:
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        target_piece = state_list[ni][nj]
                        m = str(i) + str(j) + str(ni) + str(nj)
                        if target_piece == '一一':
                            if change_state(state_list, m) != old_state_list:
                                moves.append(m)
                        elif current_player_color not in target_piece  and '炮' not in piece:
                            if '帅' in piece and '兵' in target_piece:
                                continue
                            elif '兵' in piece and '帅' in target_piece:
                                moves.append(m)
                            else:
                                if  get_strength(piece) >= get_strength(target_piece):
                                    moves.append(m)
            # 炮的合理吃法
            if '炮' in piece and current_player_color in piece:
                toY = i
                hits = False
                for toX in range(j - 1, -1, -1):
                    m = str(i) + str(j) + str(toY) + str(toX)
                    if hits is False:
                        if state_list[toY][toX] != '一一':
                            hits = True
                    else:
                        if state_list[toY][toX] != '一一':
                            if current_player_color not in state_list[toY][toX] and state_list[toY][toX] != '暗棋':
                                moves.append(m)
                            break
                hits = False
                for toX in range(j + 1, 8):
                    m = str(i) + str(j) + str(toY) + str(toX)
                    if hits is False:
                        if state_list[toY][toX] != '一一':
                            hits = True
                    else:
                        if state_list[toY][toX] != '一一':
                            if current_player_color not in state_list[toY][toX] and state_list[toY][toX] != '暗棋':
                                moves.append(m)
                            break
                toX = j
                hits = False
                for toY in range(i - 1, -1, -1):
                    m = str(i) + str(j) + str(toY) + str(toX)
                    if hits is False:
                        if state_list[toY][toX] != '一一':
                            hits = True
                    else:
                        if state_list[toY][toX] != '一一':
                            if current_player_color not in state_list[toY][toX] and state_list[toY][toX] != '暗棋':
                                moves.append(m)
                            break
                hits = False
                for toY in range(i + 1, 4):
                    m = str(i) + str(j) + str(toY) + str(toX)
                    if hits is False:
                        if state_list[toY][toX] != '一一':
                            hits = True
                    else:
                        if state_list[toY][toX] != '一一':
                            if current_player_color not in state_list[toY][toX] and state_list[toY][toX] != '暗棋':
                                moves.append(m)
                            break

    moves_id = []
    for move in moves:
        moves_id.append(move_action2move_id[move])
    return moves_id

def get_greedy_move(state_deque, current_player_color):
    state_list = state_deque[-1]
    current_color = current_player_color
    legal_moves = get_legal_moves(state_deque, current_color)

    eat_moves = []
    fallback_moves = []

    for move_id in legal_moves:
        move = move_id2move_action[move_id]
        y, x, to_y, to_x = int(move[0]), int(move[1]), int(move[2]), int(move[3])
        start_piece = state_list[y][x]
        target_piece = state_list[to_y][to_x]

        # 翻棋的情況（座標相同）
        if y == to_y and x == to_x:
            fallback_moves.append(move_id)
        elif target_piece != '一一' and current_color not in target_piece and '暗棋' not in target_piece:
            # 吃敵方棋
            eat_moves.append(move_id)
        else:
            # 普通走棋
            fallback_moves.append(move_id)

    return eat_moves,fallback_moves

# 棋盘逻辑控制
class Board(object):

    def __init__(self):
        self.last_move = None
        self._remain_backup = None
        self._reveal_backup = None
        self.remain_pieces = []
        self.first_move = True
        self.state_list = copy.deepcopy(state_list_init)
        self.game_start = False
        self.winner = None
        self.state_deque = copy.deepcopy(state_deque_init)

    # 初始化棋盘的方法
    def init_board(self, start_player = random.choice([1, 2])):   # 传入先手玩家的id
        # 增加一个颜色到id的映射字典，id到颜色的映射字
        # 永远是红方先移动

        # 初始化棋盘状态
        if CONFIG.get("no_dark_mode", True):
            self.remain_pieces = []
            self.state_list = copy.deepcopy(non_covered_state_list_init)
            self.state_deque = copy.deepcopy(non_covered_state_deque_init)
        else:
            red_pieces = ['红帅'] + ['红士'] * 2 + ['红象'] * 2 + ['红马'] * 2 + ['红车'] * 2 + ['红炮'] * 2 + ['红兵'] * 5
            black_pieces = ['黑帅'] + ['黑士'] * 2 + ['黑象'] * 2 + ['黑马'] * 2 + ['黑车'] * 2 + ['黑炮'] * 2 + ['黑兵'] * 5
            covered_pieces = red_pieces + black_pieces
            random.shuffle(covered_pieces)
            self.remain_pieces = covered_pieces
            self.state_list = copy.deepcopy(state_list_init)
            self.state_deque = copy.deepcopy(state_deque_init)
        self.first_move = True
        self.start_player = start_player
        self.id2color = {1: '红', 2: '黑'}
        self.color2id = {'红': 1, '黑': 2}
        if start_player == 1:
            self.backhand_player = 2
        else:
            self.backhand_player = 1
        # 当前手玩家，也就是先手玩家
        self.current_player_color = self.id2color[start_player]
        if start_player == 1 :
            self.current_player_id = self.color2id['红']
        else:
            self.current_player_id = self.color2id['黑']
        # 初始化最后落子位置
        self.last_move = -1
        # 记录游戏中吃子的回合数
        self.kill_action = 0
        self.game_start = False
        self.action_count = 0   # 游戏动作计数器
        self.winner = None

    def get_dark_positions(self):
        """
        回傳目前所有暗棋(未翻開)的位置列表。
        每個位置以 (row, col) 表示，例如 [(0, 3), (2, 5), ...]
        """
        dark_positions = []
        current_state = self.state_deque[-1]

        for y in range(4):  # 棋盤高
            for x in range(8):  # 棋盤寬
                if current_state[y][x] == '暗棋':
                    dark_positions.append((y, x))

        return dark_positions

    def force_reveal(self, pos, piece):
        """
        強制將某個暗棋位置翻成指定棋子（用於模擬翻棋期望）。
        不會隨機，直接覆蓋，並暫存狀態以供還原。
        """
        y, x = pos
        r, c = pos
        action = f"{r}{c}{r}{c}"
        self.last_move = action
        #print(piece,y,x,sep=" ")
        self._reveal_backup = copy.deepcopy(self.state_deque[-1])
        self._remain_backup = copy.deepcopy(self.remain_pieces)

        state_list = copy.deepcopy(self.state_deque[-1])
        state_list[y][x] = piece
        self.state_deque.append(state_list)
        if piece in self.remain_pieces:
            self.remain_pieces.remove(piece)


    def get_piece_reveal_plane(self):
        """
        回傳 shape=(32, 4, 8) 的平面：
        index 0~15 代表紅方各棋種，16~31 代表黑方。
        若該棋種仍有 n 隻未翻開，對應後 n 個 index 會填 1（代表「仍可翻出」）。
        """
        plane = np.zeros((32, 4, 8), dtype=np.float32)
        piece_index_map = {
            '红帅': [0],
            '红士': [1, 2],
            '红象': [3, 4],
            '红车': [5, 6],
            '红马': [7, 8],
            '红炮': [9, 10],
            '红兵': [11, 12, 13, 14, 15],
            '黑帅': [16],
            '黑士': [17, 18],
            '黑象': [19, 20],
            '黑车': [21, 22],
            '黑马': [23, 24],
            '黑炮': [25, 26],
            '黑兵': [27, 28, 29, 30, 31],
        }

        # 統計 remain_pieces（尚未翻出）中每種棋的數量
        remain_counts = {key: 0 for key in piece_index_map}
        for piece in self.remain_pieces:
            if piece in remain_counts:
                remain_counts[piece] += 1

        # 依照剩餘數量填 1（從後面往前填）
        for piece, count in remain_counts.items():
            idxs = piece_index_map[piece]
            for i in range(len(idxs) - count, len(idxs)):  # 填後面 count 個
                plane[idxs[i]][:, :] = 1.0

        return plane

    def get_board_str(self):
        """
        回傳目前棋盤狀態，格式為 4x8 的 list[list[str]]，
        每個元素都是棋子的字串名稱（如 '红车', '黑炮', '暗棋', '一一'）
        """
        if len(self.state_deque) == 0:
            return [["一一" for _ in range(8)] for _ in range(4)]
        return self.state_deque[-1]

    def get_history_planes(self):
        """
        將最近 4 步棋盤（含當前）轉成 shape=(7*4, 4, 8) 的張量。
        每一步棋都經過 string2array 編碼，並排列成通道方向。
        """
        history_planes = []

        # 取最近 4 步，若不足 4 步則前面補空盤
        pad_count = 4 - len(self.state_deque)
        empty_board = np.full((4, 8), '一一', dtype=object)

        for _ in range(pad_count):
            state_array = state_list2state_array(empty_board).transpose([2, 0, 1])  # [7,4,8]
            history_planes.append(state_array)

        # 把 deque 中的盤面加進來
        for state_list in self.state_deque:
            state_array = state_list2state_array(state_list).transpose([2, 0, 1])  # [7,4,8]
            history_planes.append(state_array)

        # 沿 channel (axis=0) 疊起來
        history_planes = np.concatenate(history_planes, axis=0)  # shape=(7*4,4,8)

        return history_planes

    @property
    # 获的当前盘面的所有合法走子集合
    def availables(self):
        return get_legal_moves(self.state_deque, self.current_player_color)

    def greedys(self):
        return get_greedy_move(self.state_deque, self.current_player_color)

    # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9]  CHW
    def current_state(self):
        """
        回傳目前 state 的完整特徵：
        - 前 28 層：最近 4 步棋盤，每步 7 個平面（紅黑棋子、暗棋）
        - 第 28 層：是否為先手玩家（全 1 or 全 0）
        - 第 29 層：目前視角顏色（紅方為 1，黑方為 0）
        - 後 32 層：翻棋資訊（每種棋種被翻開數量）
        共計 28 + 1 + 1 + 32 = 62 個 channel，shape = (62, 4, 8)
        """
        # Step 1: 最近 4 步棋盤（含當前）
        history_planes = self.get_history_planes()  # shape = [28, 4, 8]

        # Step 2: 先手玩家平面
        first_player_plane = np.ones((1, 4, 8), dtype=np.float32) if self.current_player_id == self.start_player \
            else np.zeros((1, 4, 8), dtype=np.float32)

        # Step 3: 當前顏色視角平面
        color_plane = np.ones((1, 4, 8), dtype=np.float32) if self.current_player_color == '红' \
            else np.zeros((1, 4, 8), dtype=np.float32)

        # Step 4: 翻棋狀態平面
        reveal_plane = self.get_piece_reveal_plane()  # shape = [32, 4, 8]

        # Step 5: 拼成最終 state
        _current_state = np.concatenate([history_planes, first_player_plane, color_plane, reveal_plane], axis=0)

        return _current_state

    def is_reveal_move(self, move):
        """只有起終點相同且該格仍覆蓋時，才是合法的翻棋動作。"""
        action = move_id2move_action.get(move)
        if action is None:
            return False
        y, x, to_y, to_x = map(int, action)
        return (y, x) == (to_y, to_x) and self.state_deque[-1][y][x] == '暗棋'

    def get_reveal_outcomes(self, move):
        """按剩餘棋種數量回傳 (棋種, 機率)，不讀取隱藏排列順序。

        保留本專案第一次翻棋必須符合當前玩家顏色的規則。
        """
        if not self.is_reveal_move(move):
            raise ValueError('Move is not a covered-square reveal')
        counts = Counter(piece for piece in self.remain_pieces
                         if not self.first_move or piece.startswith(self.current_player_color))
        total = sum(counts.values())
        if total == 0:
            raise ValueError('No eligible pieces remain for this reveal')
        return [(piece, count / total) for piece, count in counts.items()]

    # 根据move对棋盘状态做出改变
    def do_move(self, move, is_simulate=False, revealed_piece=None):
        """
        執行一步移動或翻棋，並更新棋盤狀態與當前玩家。
        
        Args:
            move: 移動的 move_id。
            is_simulate: 保留舊介面；真實與模擬翻棋使用同一個機率分布。
            revealed_piece: 指定翻棋結果，供測試或已知結果重播；必須符合規則。
            
        Returns:
            reward: 本次移動產生的獎勵（吃子得分）。
        """
        # 驗證與抽樣都先完成；無效動作／結果不能改動棋盤或回合計數。
        if move not in self.availables:
            raise ValueError('Illegal move')
        move_action = move_id2move_action[move]
        flip = False
        start_y, start_x = int(move_action[0]), int(move_action[1])
        end_y, end_x = int(move_action[2]), int(move_action[3])
        if start_x == end_x and start_y == end_y:
            flip = True
        if flip:
            outcomes = self.get_reveal_outcomes(move)
            if revealed_piece is None:
                pieces, probabilities = zip(*outcomes)
                revealed_piece = random.choices(pieces, weights=probabilities, k=1)[0]
            if revealed_piece not in dict(outcomes):
                raise ValueError('Revealed piece is not an eligible outcome')
        elif revealed_piece is not None:
            raise ValueError('Only reveal moves accept revealed_piece')
        self.game_start = True
        self.action_count += 1
        state_list = copy.deepcopy(self.state_deque[-1])
        reward = 0.0  # 新增 reward
        # 判断是否吃子
        if flip:
            self.first_move = False
            state_list[end_y][end_x] = revealed_piece
            self.remain_pieces.remove(revealed_piece)
        elif state_list[end_y][end_x] != '一一':
            reward = self.get_piece_value(state_list[end_y][end_x])
            self.kill_action = 0
        else:
            self.kill_action += 1
        # 更改棋盘状态
        if not flip:
            state_list[end_y][end_x] = state_list[start_y][start_x]
            state_list[start_y][start_x] = '一一'
        self.current_player_color = '黑' if self.current_player_color == '红' else '红'  # 改变当前玩家
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        # 记录最后一次移动的位置
        self.last_move = move
        self.state_deque.append(state_list)
        eat, fallback = self.greedys()
        if eat == [] and fallback == []:
            if self.current_player_color == '红':
                self.winner = self.color2id['黑']
            else:
                self.winner = self.color2id['红']
        return reward
    # 是否产生赢家
    def has_a_winner(self):
        """一共有三种状态，红方胜，黑方胜，平局"""
        if self.winner is not None:
            return True, self.winner
        elif self.kill_action >= CONFIG['kill_action']:  # 平局盤面分數擇優
            red_strength = self.calc_side_strength('红')
            black_strength = self.calc_side_strength('黑')
            if red_strength > black_strength:
                return True, self.color2id['红']
            elif black_strength > red_strength:
                return True, self.color2id['黑']
            else:
                return True, -1  # 平局
        return False, -1

    # 检查当前棋局是否结束
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner

        return False, -1

    def get_current_player_color(self):
        return self.current_player_color

    def get_current_player_id(self):
        return self.current_player_id

    def get_piece_value(self, piece):
        """給每個棋子一個分值"""
        values = {
            '帅': 8, '士': 5, '象': 4, '马': 2, '车': 3, '炮': 4, '兵': 1
        }
        if piece == '一一':  # 空位
            return 0
        # 去掉顏色，只保留棋子名稱
        name = piece[1:]
        return values.get(name, 0)

    def calc_side_strength(self, color):
        """計算指定顏色的棋力總分"""
        strength = 0
        for row in self.state_deque[-1]:
            for p in row:
                if p.startswith(color):
                    strength += self.get_piece_value(p)
        return strength

# 在Board类基础上定义Game类，该类用于启动并控制一整局对局的完整流程，并收集对局过程中的数据，以及进行棋盘的展示
class Game(object):

    def __init__(self, board):
        self.board = board
    # 可视化
    def graphic(self, board):
        print_board(state_list2state_array(board.state_deque[-1]))

    # 用于人机对战，人人对战等
    def start_play(self, player1, player2, start_player=1, is_shown=1):
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 (player1 first) '
                            'or 2 (player2 first)')
        self.board.init_board(start_player)  # 初始化棋盘
        if start_player==1:
            p1, p2 = 1, 2
            player1.set_player_ind(1)
            player2.set_player_ind(2)
            players = {p1: player1, p2: player2}
        else:
            p1, p2 = 1, 2
            player1.set_player_ind(1)
            player2.set_player_ind(2)
            players = {p1: player1, p2: player2}

        if is_shown:
            self.graphic(self.board)
        while True:
            current_player = self.board.get_current_player_id()  # 红子对应的玩家id
            player_in_turn = players[current_player]  # 决定当前玩家的代理
            move = player_in_turn.get_action(self.board)
            state = self.board.current_state()
            # print(self.board.remain_pieces)
            self.board.do_move(move)  # 棋盘做出改变

            if is_shown:
                self.graphic(self.board)
                red_strength = self.board.calc_side_strength('红')
                black_strength = self.board.calc_side_strength('黑')
                if red_strength != 0 and black_strength != 0:
                    print("紅 : ", red_strength, sep="")
                    print("黑 : ", black_strength, sep="")
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner]," ",self.board.current_player_color,sep="")
                else:
                    print("Game end. Tie")
                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=False, temp=1):
        self.board.init_board()     # 初始化棋盘, start_player=1
        p1, p2 = 1, 2
        states, mcts_probs, current_players = [], [], []
        # 开始自我对弈
        _count = 0
        eat_count = 0
        rewards = []
        while True:
            _count += 1
            if is_shown:
                self.graphic(self.board)
            move, move_probs = player.get_action(
                self.board,
                temp=temp,
                return_prob=True
            )
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player_id)
            # 保存自我对弈的数据
            # 轉換成 numpy array

            # 执行一步落子
            reward = self.board.do_move(move)
            # 翻棋結果已知後，才可沿用對應的搜尋子樹。
            observe_move = getattr(player, 'observe_move', None)
            if observe_move is not None:
                observe_move(self.board, move)
            if reward != 0:
                rewards.append(reward)
            else:
                rewards.append(0.0)
            end, winner = self.board.game_end()
            if end:
                # 先生成勝負結果 (winner_z)
                # 1. 生成 winner_z (主導)
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = -1.0

                # 2. 吃子分數正規化 (副作用)
                total_abs = sum(abs(r) for r in rewards) + 1e-8
                scaled_rewards = np.array([r / total_abs for r in rewards])  # [-1, 1] 之間
                alpha = 5  # 吃子影響比重 (可以調整 0.1 ~ 0.5)

                # 3. 合併，勝負為主，吃子為輔
                merged_rewards = winner_z + alpha * scaled_rewards

                # 4. 縮放到 [-1, 1]
                max_abs = np.max(np.abs(merged_rewards)) + 1e-8
                merged_rewards = merged_rewards / max_abs

                winner_z = merged_rewards  # 取代掉原本的 winner_z

                red_rewards = [r for r, p in zip(winner_z, current_players) if p == 1]
                black_rewards = [r for r, p in zip(winner_z, current_players) if p == 2]

                print("Red rewards:", red_rewards)
                print("Black rewards:", black_rewards)

                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is:", winner)
                    else:
                        print('Game end. Tie')

                return winner, zip(states, mcts_probs, winner_z)

if __name__ == '__main__':
    board = Board()



