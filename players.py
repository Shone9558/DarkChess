# players.py
import copy
import random

from matplotlib.ticker import MaxNLocator

from game import Game, state_list2state_array, array2string, string2array, move_id2move_action, move_action2move_id, Board
from mcts import MCTSPlayer
import csv
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from openai import OpenAI

from pytorch_net import PolicyValueNet

# 設定中文字體（以 Windows 為例）
rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
rcParams['axes.unicode_minus'] = False


class MinimaxDarkChessPlayer:
    """暗棋 Alpha–Beta + 向上傳遞 + 寧靜搜尋 AI 玩家"""

    def __init__(self, search_depth=4):
        self.game = None
        self.depth = search_depth
        self.player = None
        self.agent = "AI"

    def set_player_ind(self, p):
        """設定玩家編號：1=紅, 2=黑"""
        self.player = p

    # ====================== 改良版搜尋 ======================
    def quiescence_search(self, board, alpha, beta, side_player_id, maxDepth, depth):
        """寧靜搜尋：只展開吃子直到穩定。"""
        eat_moves, _ = board.greedys()
        if not eat_moves:
            return 0

        best = -float("inf")
        for move in eat_moves:
            backup = self._backup_board(board)
            reward = board.do_move(move)  # reward 為吃子分數（正代表吃對方）
            # value = 該吃子分數 + 同層深度 bonus
            bonus = (maxDepth - depth)
            value = reward + bonus
            # 遞迴吃到底
            value -= self.quiescence_search(board, -beta, -alpha, side_player_id, maxDepth, depth + 1)
            self._restore_board(board, backup)

            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def search_upward(self, board, depth, alpha, beta, side_player_id, maxDepth):
        """向上傳遞 + 同分步處理 + 單向搜尋 (negamax style)"""
        # --- 終止條件 ---
        if depth == 0:
            eat_moves, _ = board.greedys()
            if eat_moves:
                return self.quiescence_search(board, alpha, beta, side_player_id, maxDepth, 0)
            return 0

        eat_moves, fallback_moves = board.greedys()
        legal_moves = eat_moves + fallback_moves
        if not legal_moves:
            return 0

        value_accum = -float("inf")
        all_zero_scores = True

        for move in legal_moves:
            backup = self._backup_board(board)
            reward = board.do_move(move)  # reward 為吃子加權值（正：吃對手，負：被吃）
            # 初始分值：若有吃子則加上同分步bonus
            value_here = 0
            if reward != 0:
                bonus = (maxDepth - depth)
                value_here += reward + bonus
                child_depth = depth if depth == maxDepth - 1 else depth - 1
                value_here -= self.search_upward(board, child_depth, -beta, -alpha, side_player_id, maxDepth)
            else:
                value_here -= self.search_upward(board, depth - 1, -beta, -alpha, side_player_id, maxDepth)

            self._restore_board(board, backup)

            if value_here > value_accum:
                value_accum = value_here
            if value_accum > alpha:
                alpha = value_accum
            if alpha >= beta:
                break

            if value_here != 0:
                all_zero_scores = False

        # --- 單向搜尋（當所有候選皆為 0） ---
        if all_zero_scores and depth > 1:
            one_way_best = -float("inf")
            eat_moves2, fallback_moves2 = board.greedys()
            for move in eat_moves2 + fallback_moves2:
                backup = self._backup_board(board)
                reward = board.do_move(move)
                val = reward - self.search_upward(board, depth - 1, -beta, -alpha, side_player_id, maxDepth)
                self._restore_board(board, backup)
                if val > one_way_best:
                    one_way_best = val
                if one_way_best > alpha:
                    alpha = one_way_best
                if alpha >= beta:
                    break
            if one_way_best > value_accum:
                value_accum = one_way_best

        return value_accum

    # ====================== 動作選擇 ======================
    def get_action(self, board):
        """取得下一步行動"""
        self.game = Game(board)
        eat_moves, fallback_moves = board.greedys()
        legal_moves = eat_moves + fallback_moves
        if not legal_moves:
            return None

        # 有吃子 → 用新搜尋法
        if eat_moves:
            best_value = -float("inf")
            best_move = None
            alpha, beta = -float("inf"), float("inf")
            side_player_id = board.current_player_id
            maxDepth = self.depth

            for move in eat_moves:
                backup = self._backup_board(board)
                reward = board.do_move(move)
                value_here = reward
                if reward != 0:
                    bonus = (maxDepth - 1)
                    value_here += bonus
                    value_here -= self.search_upward(board, maxDepth - 1, -beta, -alpha, side_player_id, maxDepth)
                else:
                    value_here -= self.search_upward(board, maxDepth - 1, -beta, -alpha, side_player_id, maxDepth)
                self._restore_board(board, backup)

                if value_here > best_value:
                    best_value = value_here
                    best_move = move

            return best_move
        else:
            # 沒吃子 → 翻子策略
            return self._reveal_strategy(board)

        # ================== 翻子策略 ==================

    def _reveal_strategy(self, board):
        """模擬翻子期望值，挑出最有利的翻子位置"""
        dark_positions = board.get_dark_positions()
        if not dark_positions:
            return random.choice(board.availables)  # 沒得翻就隨便走

        best_pos = None
        best_value = -float('inf')

        for pos in dark_positions:
            expected_value = self._simulate_reveal(board, pos)
            if expected_value > best_value:
                best_value = expected_value
                best_pos = pos

        r, c = best_pos
        action = f"{r}{c}{r}{c}"
        return move_action2move_id[action]

    def _simulate_reveal(self, board, pos):
        """模擬翻出不同棋子後的期望值（結合新搜尋法）"""
        re_pieces = board.remain_pieces
        if not re_pieces:
            return 0

        # 統計棋種機率
        counts = {}
        for p in re_pieces:
            counts[p] = counts.get(p, 0) + 1
        total = len(re_pieces)
        possible_pieces = {p: c / total for p, c in counts.items()}

        total_value = 0
        for piece, prob in possible_pieces.items():
            backup = self._backup_board(board)
            board.force_reveal(pos, piece)

            # 🔄 改成呼叫新搜尋器 search_upward
            score = self.search_upward(
                board, depth=2,
                alpha=-float('inf'),
                beta=float('inf'),
                side_player_id=board.current_player_id,
                maxDepth=2
            )

            total_value += prob * score
            self._restore_board(board, backup)

        return total_value

    # ====================== 評估與備份 ======================
    def evaluate(self, board, winner=None):
        """盤面評估"""
        if winner == self.player:
            return 9999
        elif winner != -1 and winner != self.player:
            return -9999

        red_strength = board.calc_side_strength('红')
        black_strength = board.calc_side_strength('黑')
        return red_strength - black_strength if self.player == 1 else black_strength - red_strength

    def _backup_board(self, board):
        """備份棋盤狀態"""
        return {
            "state_deque": copy.deepcopy(board.state_deque),
            "remain_pieces": copy.deepcopy(board.remain_pieces),
            "current_player_color": board.current_player_color,
            "current_player_id": board.current_player_id,
            "last_move": board.last_move,
            "winner": board.winner,
            "kill_action": board.kill_action,
            "first_move": board.first_move,
            "action_count": board.action_count,
        }

    def _restore_board(self, board, backup):
        """還原棋盤狀態"""
        board.state_deque = backup["state_deque"]
        board.remain_pieces = backup["remain_pieces"]
        board.current_player_color = backup["current_player_color"]
        board.current_player_id = backup["current_player_id"]
        board.last_move = backup["last_move"]
        board.winner = backup["winner"]
        board.kill_action = backup["kill_action"]
        board.first_move = backup["first_move"]
        board.action_count = backup["action_count"]


class ChatGPTPlayer:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key="")  # 放 API Key

        self.model = model
        self.agent = 'AI'

    def set_player_ind(self, p):
        self.player = p

    def board_to_state(self, _state_array):
        # _state_array: [10, 9, 7], HWC
        all_board = []

        for i in range(4):
            board_line = []
            for j in range(8):
                board_line.append(array2string(_state_array[i][j]))
            all_board.append(board_line)
        return all_board

    def get_action(self, board):
        # 轉換棋盤
        state_text = self.board_to_state(state_list2state_array(board.state_deque[-1]))

        # 把所有合法行動列出來
        eat_move_list, fallback_movelist = board.greedys()
        all_moves = eat_move_list + fallback_movelist
        all_moves_relabel = [move_id2move_action[m] for m in all_moves]
        # 生成 prompt
        # print(all_moves_relabel)
        prompt = f"""
你正在玩中國暗棋 (4x8)。
現在棋盤狀態如下：
{state_text}

可行的動作有：
{all_moves_relabel} 
格式是4個數字 ABCD AB為起始位置 CD為結束位置 若A=C B=D則代表翻棋

請從中選擇一個最佳動作，直接輸出該動作 (不要多餘的解釋)。
"""

        try:
            # 呼叫 ChatGPT
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )
            reply = response.choices[0].message.content.strip()
            reply_str = "".join(str(x) for x in reply if x.isdigit())  # 過濾非數字

            # 嘗試轉換
            if reply_str in move_action2move_id:
                reply_index = move_action2move_id[reply_str]
                # 確保動作合法
                if reply_index in all_moves:
                    return reply_index
                else:
                    print(f"⚠️ ChatGPT 回傳非法動作 {reply_str}（不在合法動作列表），改用隨機動作")
            else:
                print(f"⚠️ ChatGPT 回傳無效動作字串 {reply}，改用隨機動作")

        except Exception as e:
            print(f"⚠️ ChatGPT API 錯誤: {e}，改用隨機動作")

            # fallback
        return random.choice(all_moves) if all_moves else None

class RandomPlayer:
    def __init__(self):
        self.agent = 'AI'
        self.type = "random"

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        eat_move_list, fallback_move_list = board.greedys()
        all_move = eat_move_list + fallback_move_list
        if not all_move:
            return None
        return random.choice(all_move)


class GreedyPlayer:
    def __init__(self):
        self.agent = 'AI'
        self.type = "greedy"

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        eat_move_list, fallback_move_list = board.greedys()
        if eat_move_list:
            return random.choice(eat_move_list)
        elif fallback_move_list:
            return random.choice(fallback_move_list)
        else:
            return None


class Human:
    def __init__(self):
        self.agent = 'Human'

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        # UIplay 會用，這裡先放佔位
        return None



def battle_summary(player1, opponents, board, playouts, n_games=100,
                   save_csv=True, csv_file="battle_summary.csv"):
    """
    對每個對手進行對戰並記錄結果。
    若 CSV 已存在，則只會追加新對手的結果（不覆蓋舊資料）。
    """
    game = Game(board)
    results = {}

    existing_opponents = set()
    writer = None
    f = None

    # --- 檢查是否已有結果檔 ---
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f_read:
            reader = csv.reader(f_read)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 1:
                    existing_opponents.add(row[0])

    # --- 開啟 CSV 檔案 ---
    if save_csv:
        file_exists = os.path.exists(csv_file)
        f = open(csv_file, "a", newline="", encoding="utf-8")
        writer = csv.writer(f)
        # 如果是新檔案就寫入標頭
        if not file_exists:
            writer.writerow(["Opponent", "Wins", "Losses", "Draws"])

    # --- 逐一對戰 ---
    for opp_name, player2 in opponents.items():
        if opp_name in existing_opponents:
            print(f"⚠️ {opp_name} 已存在於 {csv_file}，跳過。")
            continue

        stats = {"win": 0, "loss": 0, "draw": 0}
        print(f"⚔️ {player1.agent} vs {opp_name} 開始對戰，共 {n_games} 場...")

        for i in range(n_games):
            board.init_board(1)
            players = {1: player1, 2: player2}

            # 輪流先手
            if i % 2 == 0:
                player1.set_player_ind(1)
                player2.set_player_ind(2)
            else:
                player1.set_player_ind(2)
                player2.set_player_ind(1)

            while True:
                move = players[board.current_player_id].get_action(board)
                if move is None:
                    break
                board.do_move(move)
                end, winner = board.game_end()
                if end:
                    print(f"第{i}場結束")
                    if winner == -1:
                        stats["draw"] += 1
                    elif players[winner] == player1:
                        stats["win"] += 1
                    else:
                        stats["loss"] += 1
                    break

        results[opp_name] = stats
        print(f"✅ {player1.agent} vs {opp_name} 完成: {stats}")

        # --- 寫入結果 ---
        if writer:
            writer.writerow([opp_name, stats["win"], stats["loss"], stats["draw"]])
            f.flush()

    if f:
        f.close()
        print(f"📊 對戰結果已更新至 {csv_file}")

    return results

def plot_battle_results_from_csv(csv_file="battle_summary.csv"):
    import csv
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.transforms import blended_transform_factory

    opponents = []
    wins = []
    draws = []
    losses = []

    # --- 讀取 CSV ---
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opponents.append(row.get("Opponent", "").strip())

            def to_int(x):
                try:
                    return int(x)
                except:
                    return 0

            wins.append(to_int(row.get("Wins", 0)))
            losses.append(to_int(row.get("Losses", 0)))
            draws.append(to_int(row.get("Draws", 0)))

    if not opponents:
        print("CSV 沒有讀到資料")
        return

    # --- 依 Wins 排序 ---
    order = sorted(range(len(opponents)), key=lambda i: wins[i], reverse=False)
    opponents = [opponents[i] for i in order]
    wins = [wins[i] for i in order]
    draws = [draws[i] for i in order]
    losses = [losses[i] for i in order]

    # --- 顏色 ---
    color_win = "#4A90E2"   # 藍
    color_draw = "#7F8C8D"  # 灰
    color_loss = "#E74C3C"  # 紅

    y = np.arange(len(opponents))
    height = 0.6

    # 調整 figsize 與左邊邊界（避免左側文字被裁切）
    fig, ax = plt.subplots(figsize=(5.5, max(4, len(opponents) * 0.6)))
    fig.subplots_adjust(left=0.20, right=0.95)  # left 越大，軸外可放的空間越多

    # Bars（從 x=0 開始）
    win_bars = ax.barh(y, wins, height=height, color=color_win, edgecolor="black")
    draw_bars = ax.barh(y, draws, height=height, left=wins, color=color_draw, edgecolor="black")
    left_for_losses = [w + d for w, d in zip(wins, draws)]
    loss_bars = ax.barh(y, losses, height=height, left=left_for_losses, color=color_loss, edgecolor="black")

    # 標註每個區塊數字（>0 才顯示）
    def annotate(bars):
        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(bar.get_x() + w / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(w)),
                        va="center", ha="center",
                        fontsize=9,
                        color="white" if w > 10 else "black")
    annotate(win_bars)
    annotate(draw_bars)
    annotate(loss_bars)

    # --- 把對手名稱顯示在右邊（跟之前行為一致）---
    ax.yaxis.tick_right()
    ax.set_yticks(y)
    ax.set_yticklabels(opponents, fontsize=11)

    # --- 在 bar 左側軸外放置 "暗棋 Alpha"（x 用 axes fraction，y 用 data）---
    # blended transform: (x in axes coords, y in data coords)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    # 放在 axes fraction 的 -0.02（略在軸外），若要更外可調到 -0.04 或 -0.01
    x_axes_pos = -0.02
    for yi in y:
        ax.text(x_axes_pos, yi, "暗棋阿拉法",
                transform=trans,
                ha="right", va="center", fontsize=11,
                clip_on=False)  # 允許畫在軸外

    # 標題與格式
    ax.set_title("暗棋阿拉法對戰結果", fontsize=15, weight="bold")
    ax.set_xlabel("場數", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # x 軸上界（從 0 開始）
    total_max = max(w + d + l for w, d, l in zip(wins, draws, losses))
    ax.set_xlim(0, total_max + max(5, int(total_max * 0.05)))

    # 移除圖例（你先前要求）
    # （若你之後想要圖例，可反註解下面三行）
    # from matplotlib.patches import Patch
    # ax.legend(handles=[Patch(facecolor=color_win), Patch(facecolor=color_draw), Patch(facecolor=color_loss)], labels=["Wins","Draws","Losses"])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    plot_battle_results_from_csv()