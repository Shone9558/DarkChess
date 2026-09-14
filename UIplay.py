import pygame
import sys
import copy
import random
import numpy as np
from game import move_action2move_id, Game, Board, move_id2move_action
from mcts import MCTSPlayer
import matplotlib.pyplot as plt
import time
from config import CONFIG
from mcts_pure import MCTS_Pure
from players import GreedyPlayer, RandomPlayer, ChatGPTPlayer, MinimaxDarkChessPlayer
if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')

class Human:

    def __init__(self):
        self.agent = 'HUMAN'

    def get_action(self, move):
        # move从鼠标点击事件触发
        # print('当前是player2在操作')
        # print(board.current_player_color)
        if  move_action2move_id.__contains__(move):
            move = move_action2move_id[move]
        else:
            move = -1
        # move = random.choice(board.availables)
        return move

    def set_player_ind(self, p):
        self.player = p


if CONFIG['use_frame'] == 'paddle':
    policy_value_net = PolicyValueNet(model_file='current_policy.model')
elif CONFIG['use_frame'] == 'pytorch':
    policy_value_net = PolicyValueNet(model_file='new_current_policy.pth')

    #policy_value_net = PolicyValueNet(model_file=None)
else:
    print('暂不支持您选择的框架')

# 初始化pygame
pygame.init()
pygame.mixer.init()
size = width, height = 700, 700
bg_image = pygame.image.load('imgs/board.png')  #图片位置
bg_image = pygame.transform.smoothscale(bg_image, size)

clock = pygame.time.Clock()
fullscreen = False
# 创建指定大小的窗口
screen = pygame.display.set_mode(size)
# 设置窗口标题
pygame.display.set_caption("暗棋")

# 加载一个列表进行图像的绘制
# 列表表示的棋盘初始化，红子在上，黑子在下，禁止对该列表进行编辑，使用时必须使用深拷贝

board_list_init = []
for i in range(4):
    row = []
    for j in range(8):
        row.append('暗棋')
    board_list_init.append(row)

# 加载棋子被选中的图片
fire_image = pygame.transform.smoothscale(pygame.image.load("imgs/fire.png").convert_alpha(), (width // 10, height // 10))
fire_image.set_alpha(200)

# 制作一个从字符串到pygame.surface对象的映射
str2image = {
    '暗棋': pygame.transform.smoothscale(pygame.image.load("imgs/blankchess.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红车': pygame.transform.smoothscale(pygame.image.load("imgs/hongche.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红马': pygame.transform.smoothscale(pygame.image.load("imgs/hongma.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红象': pygame.transform.smoothscale(pygame.image.load("imgs/hongxiang.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红士': pygame.transform.smoothscale(pygame.image.load("imgs/hongshi.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红帅': pygame.transform.smoothscale(pygame.image.load("imgs/hongshuai.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红炮': pygame.transform.smoothscale(pygame.image.load("imgs/hongpao.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红兵': pygame.transform.smoothscale(pygame.image.load("imgs/hongbing.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑车': pygame.transform.smoothscale(pygame.image.load("imgs/heiche.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑马': pygame.transform.smoothscale(pygame.image.load("imgs/heima.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑象': pygame.transform.smoothscale(pygame.image.load("imgs/heixiang.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑士': pygame.transform.smoothscale(pygame.image.load("imgs/heishi.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑帅': pygame.transform.smoothscale(pygame.image.load("imgs/heishuai.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑炮': pygame.transform.smoothscale(pygame.image.load("imgs/heipao.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑兵': pygame.transform.smoothscale(pygame.image.load("imgs/heibing.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
}
str2image_rect = {
    '暗棋': str2image['暗棋'].get_rect(),
    '红车': str2image['红车'].get_rect(),
    '红马': str2image['红马'].get_rect(),
    '红象': str2image['红象'].get_rect(),
    '红士': str2image['红士'].get_rect(),
    '红帅': str2image['红帅'].get_rect(),
    '红炮': str2image['红炮'].get_rect(),
    '红兵': str2image['红兵'].get_rect(),
    '黑车': str2image['黑车'].get_rect(),
    '黑马': str2image['黑马'].get_rect(),
    '黑象': str2image['黑象'].get_rect(),
    '黑士': str2image['黑士'].get_rect(),
    '黑帅': str2image['黑帅'].get_rect(),
    '黑炮': str2image['黑炮'].get_rect(),
    '黑兵': str2image['黑兵'].get_rect()
}

str2image_selected = {}
for key, img in str2image.items():
    w, h = img.get_size()
    scale_img = pygame.transform.smoothscale(img, (int(w * 1.15), int(h * 1.15)))  # 放大 1.2 倍
    str2image_selected[key] = scale_img
# 根据棋盘列表获得最新位置
# 返回一个由image和rect对象组成的列表
x_ratio = 80
y_ratio = 72
x_bais = 70
y_bais = 65
def board2image(board):
    return_image_rect = []
    for i in range(4):
        for j in range(8):
            piece = board[i][j]
            if piece not in str2image:
                continue

            is_selected = 'start_i_j' in globals() and (j, i) == start_i_j

            # 中心位置
            center_x = j * x_ratio + x_bais
            center_y = i * y_ratio + y_bais

            if is_selected:
                img = str2image_selected[piece]
            else:
                img = str2image[piece]

            rect = img.get_rect()
            rect.center = (center_x, center_y)
            return_image_rect.append((img, rect))
    return return_image_rect

def show_current_state(state, mode="auto", num_planes_per_group=9):
    """
    Display all planes of the dark chess state encoding using numeric values.
    Each cell shows its value directly (no color map).

    state: shape (62, 4, 8)
    mode: "auto" (grouped) or "manual" (show all)
    num_planes_per_group: number of planes per group when in manual mode
    """
    state = np.array(state)
    assert len(state.shape) == 3, f"Invalid input shape: {state.shape}"
    total_planes, h, w = state.shape

    # === Define logical groups ===
    groups = [
        ("Board History (last 4 moves × 7 planes = 28)", 0, 28),
        ("Player and Perspective Info (2 planes)", 28, 30),
        ("Piece Reveal States (32 planes)", 30, 62)
    ]

    def show_group(title, start, end):
        planes_to_show = state[start:end]
        num_show = planes_to_show.shape[0]

        cols = int(np.ceil(np.sqrt(num_show)))
        rows = int(np.ceil(num_show / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes = np.array(axes).reshape(-1)

        for i in range(rows * cols):
            ax = axes[i]
            if i < num_show:
                plane = planes_to_show[i]
                ax.set_xlim(-0.5, w - 0.5)
                ax.set_ylim(h - 0.5, -0.5)

                # Draw grid
                ax.set_xticks(np.arange(-0.5, w, 1))
                ax.set_yticks(np.arange(-0.5, h, 1))
                ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.4)

                # Show numeric values
                for y in range(h):
                    for x in range(w):
                        val = plane[y, x]
                        ax.text(x, y, f"{val:.0f}", ha='center', va='center', fontsize=10)

                ax.set_title(f"Plane {start + i}", fontsize=10)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                ax.axis('off')

        plt.suptitle(f"{title} [{start}–{end - 1}]  shape={state.shape}", fontsize=13)
        plt.tight_layout()
        plt.show()

    if mode == "auto":
        for title, start, end in groups:
            show_group(title, start, end)
    else:
        for start in range(0, total_planes, num_planes_per_group):
            end = min(start + num_planes_per_group, total_planes)
            show_group(f"Planes {start}–{end - 1}", start, end)


fire_rect = fire_image.get_rect()
fire_rect.center = (0 * x_ratio + x_bais, 3 * y_ratio + y_bais)

# 加载两个玩家，AI对AI，或者AI对human
board=Board()
# 开始的玩家
start_player = 1

player_human = Human()

player_random = MCTS_Pure(500)

player_dark_craft = MinimaxDarkChessPlayer()

player_RL = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=3,
                                 n_playout=500,
                                 is_selfplay=0)
player_RL2 = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=3,
                                 n_playout=500,
                                 is_selfplay=0)
player_greedy = GreedyPlayer()

# player_gpt = ChatGPTPlayer() #更動
# player2 = Human()

board.init_board(start_player)
p1, p2 = 1, 2
player_RL.set_player_ind(1)
player_RL2.set_player_ind(2)
# player_gpt.set_player_ind(2) #更動
players = {p1: player_RL, p2: player_RL2}


# 切换玩家
swicth_player = True
draw_fire = False
move_action = ''
first_button = False
while True:

    # 填充背景
    screen.blit(bg_image, (0, 0))
    for image, image_rect in board2image(board=board.state_deque[-1]):
        screen.blit(image, image_rect)
    if draw_fire:
        screen.blit(fire_image, fire_rect)
    # 更新界面
    pygame.display.update()
    # 不高于60帧
    clock.tick(60)
    player_color = board.current_player_color
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:    #按下按键
            # print("[MOUSEBUTTONDOWN]", event.pos, event.button)
            mouse_x, mouse_y = event.pos
            if not first_button:
                for i in range(4):
                    for j in range(8):
                        if abs(80 * j + 70 - mouse_x) < 30 and abs(72 * i + 70 - mouse_y) < 30:
                            piece_name = board.state_deque[-1][i][j]
                            if piece_name != '暗棋' and piece_name != '':
                                if player_color == '红' and not piece_name.startswith('红'):
                                    continue  # 紅方玩家不能點黑棋
                                if player_color == '黑' and not piece_name.startswith('黑'):
                                    continue  # 黑方玩家不能點紅棋
                            first_button = True
                            start_i_j = j, i
                            fire_rect.center = (start_i_j[0] * x_ratio + x_bais, start_i_j[1] * y_ratio + y_bais)
                            # print(start_i_j)
                            # screen.blit(fire_image, fire_rect)
                            break

            elif first_button:
                for i in range(4):
                    for j in range(8):
                        if abs(80 * j + 70 - mouse_x) < 30 and abs(72 * i + 70 - mouse_y) < 30:
                            first_button = False
                            end_i_j = j, i
                            move_action = str(start_i_j[1]) + str(start_i_j[0]) + str(end_i_j[1]) + str(end_i_j[0])
                            # screen.blit(fire_image, fire_rect)

    if swicth_player:
        current_player = board.get_current_player_id()  # 红子对应的玩家id
        player_in_turn = players[current_player]  # 决定当前玩家的代理

    if player_in_turn.agent != 'HUMAN':
        pygame.display.update()
        start_time = time.time()
        move = player_in_turn.get_action(board)  # 当前玩家代理拿到动作
        state = board.current_state()
        
        #print(board.remain_pieces)
        #show_current_state(state, mode="auto")
        '''
        state = np.expand_dims(state, 0)  # 增加 batch 維度
        state = state.astype('float32')
        _, v = policy_value_net.policy_value(state)
        # 列出所有合法動作及其概率
        acts = board.availables  # 合法動作的全域 ID
        print("合法動作數量:", len(acts))
        print("合法動作及機率分佈:")

        for move_id in acts:
            prob = probs[move_id]
            y1, x1, y2, x2 = map(int, move_id2move_action[move_id])
            print(f"Move {move_id}: ({y1},{x1})->({y2},{x2}), Prob = {prob:.4f}")

        # 取出合法動作裡的 top3
        legal_probs = np.array([probs[a] for a in acts])
        top3_idx = np.argsort(legal_probs)[-3:][::-1]
        print("\n🔝 Top 3 legal moves:")
        for rank, i in enumerate(top3_idx, 1):
            move_id = acts[i]
            y1, x1, y2, x2 = map(int, move_id2move_action[move_id])
            print(f"{rank}. Move {move_id}: ({y1},{x1})->({y2},{x2}), Prob = {legal_probs[i]:.4f}")

        print(f"\nPredicted V for Player {player_in_turn}: {v}")
        '''
        print('耗时：', time.time() - start_time)
        board.do_move(move)  # 棋盘做出改变
        swicth_player = True
        if 'start_i_j' in globals():
            del start_i_j
    elif player_in_turn.agent == 'HUMAN':
        swicth_player = False
        if len(move_action) == 4:
            move = player_in_turn.get_action(move_action)  # 人類從UI滑鼠操作產生的move_action
            if move != -1 and move in board.availables:
                board.do_move(move)
                swicth_player = True
                move_action = ''
                if 'start_i_j' in globals():
                    del start_i_j

    end, winner = board.game_end()
    if end:
        if winner != -1:
            print("Game end. Winner is", players[winner])
        else:
            print("Game end. Tie")
        sys.exit()