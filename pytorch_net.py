import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== SE Block（針對 62 通道優化）====================
class SEBlock(nn.Module):
    """針對暗棋 62 通道優化的 SE Block"""

    def __init__(self, channels=62, reduction=8):
        super().__init__()
        # 確保降維後至少有 4 個神經元
        reduced_channels = max(channels // reduction, 4)

        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)

    def forward(self, x):
        b, c, h, w = x.size()

        # Squeeze: 全局平均池化
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Scale: 重新加權
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==================== 改進的殘差塊 ====================
class ResBlock(nn.Module):
    def __init__(self, num_filters=256, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.use_se = use_se
        if use_se:
            # 這裡的 SE 是作用在殘差塊的輸出上（256 通道）
            self.se = SEBlock(num_filters, reduction=16)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.use_se:
            y = self.se(y)

        return F.relu(x + y)


# ==================== 針對你的 State 設計的完整網路 ====================
class DarkChessNet(nn.Module):
    def __init__(self, num_channels=256, num_res_blocks=10, num_actions=384):
        super().__init__()

        # ⭐ 輸入：[B, 62, 4, 8] - 來自你的 Board.current_state()
        self.input_channels = 62

        # 第一層卷積：62 -> 256 通道
        self.conv_block = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv_block_bn = nn.BatchNorm2d(num_channels)

        # ⭐ 可選：在輸入層後加一個 SE Block（針對 62 通道）
        self.input_se = SEBlock(self.input_channels, reduction=8)

        # 殘差塊（帶 SE）
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels, use_se=True)
            for _ in range(num_res_blocks)
        ])

        # --- 策略頭 ---
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc1 = nn.Linear(32 * 4 * 8, 512)
        self.policy_fc2 = nn.Linear(512, num_actions)
        self.policy_dropout = nn.Dropout(0.2)

        # --- 價值頭 ---
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 4 * 8, 512)
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: [B, 62, 4, 8]

        # ⭐ 可選：先對輸入的 62 通道做 SE
        x = self.input_se(x)

        # 第一層卷積
        x = F.relu(self.conv_block_bn(self.conv_block(x)))
        # x shape: [B, 256, 4, 8]

        # 殘差塊
        for res in self.res_blocks:
            x = res(x)

        # --- 策略頭 ---
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 32 * 4 * 8)
        p = F.relu(self.policy_fc1(p))
        p = self.policy_dropout(p)
        p = self.policy_fc2(p)
        p = F.log_softmax(p, dim=1)

        # --- 價值頭 ---
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 16 * 4 * 8)
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout(v)
        v = F.relu(self.value_fc2(v))
        v = torch.tanh(self.value_fc3(v))

        return p, v


# ==================== 與你的 Board 整合測試 ====================
class PolicyValueNet:
    def __init__(self, model_file=None, use_gpu=True, device=None):
        # 自動判斷：如果沒有 GPU，會改用 CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.use_gpu = use_gpu and (device == "cuda" and torch.cuda.is_available())
        self.device = torch.device(device)
        print(f"[PolicyValueNet] Using device: {self.device}")

        self.l2_const = 1e-4

        # 使用新的網路
        self.policy_value_net = DarkChessNet(
            num_channels=256,
            num_res_blocks=10,
            num_actions=384
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.policy_value_net.parameters(),
            lr=2e-3,
            weight_decay=self.l2_const
        )

        if model_file:
            # 加上 map_location，讓在 CPU 也能載入 GPU 訓練好的模型
            self.policy_value_net.load_state_dict(
                torch.load(model_file, map_location=self.device)
            )

    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = torch.exp(log_act_probs).cpu().numpy()
        return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """與你的 Board 類整合"""
        self.policy_value_net.eval()

        # ⭐ 使用你的 Board.current_state() 方法
        # 這會返回 shape=(62, 4, 8) 的 numpy array
        current_state = board.current_state()

        # 轉換成 PyTorch tensor
        current_state = np.ascontiguousarray(
            current_state.reshape(1, 62, 4, 8)
        ).astype('float32')
        current_state = torch.as_tensor(current_state).to(self.device)

        # 前向傳播
        with torch.amp.autocast("cuda"):
            log_act_probs, value = self.policy_value_net(current_state)

        # 轉回 CPU 並處理合法動作
        log_act_probs = log_act_probs.cpu()
        value = value.cpu()

        act_probs = np.exp(log_act_probs.detach().numpy().flatten())
        legal_positions = board.availables
        act_probs = zip(legal_positions, act_probs[legal_positions])

        return act_probs, value.detach().numpy()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=None):
        """訓練一步"""
        self.policy_value_net.train()

        # state_batch shape: [B, 62, 4, 8]
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        mcts_probs = torch.tensor(mcts_probs, dtype=torch.float32).to(self.device)
        winner_batch = torch.tensor(winner_batch, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()

        if lr is not None:
            for params in self.optimizer.param_groups:
                params['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        value = value.view(-1)

        # 損失函數
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        loss = value_loss + policy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )

        return loss.item(), entropy.item()

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)


# ==================== 測試與你的 State 相容性 ====================
if __name__ == '__main__':
    print("=" * 70)
    print("測試與暗棋 Board 的整合")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")

    # 創建網路
    net = DarkChessNet(num_channels=256, num_res_blocks=10).to(device)

    # 模擬你的 Board.current_state() 輸出
    # shape = (62, 4, 8)
    test_state = np.random.randn(62, 4, 8).astype('float32')

    # 轉換成 batch 格式
    test_batch = torch.tensor(test_state).unsqueeze(0).to(device)
    print(f"\n輸入 shape: {test_batch.shape}")  # [1, 62, 4, 8]

    # 前向傳播
    with torch.no_grad():
        log_probs, value = net(test_batch)

    print(f"策略輸出 shape: {log_probs.shape}")  # [1, 384]
    print(f"價值輸出 shape: {value.shape}")  # [1, 1]

    # 參數統計
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\n總參數量: {total_params:,}")

    # SE Block 參數
    se_params = sum(
        p.numel() for n, p in net.named_parameters()
        if 'se' in n or 'input_se' in n
    )
    print(f"SE Block 參數: {se_params:,} ({se_params / total_params * 100:.2f}%)")

    print("\n" + "=" * 70)
    print("✅ 測試通過！網路可以正常處理你的 State 格式")
    print("=" * 70)

    # 顯示 SE Block 配置
    print("\nSE Block 配置:")
    print(f"  - 輸入層 SE: 62 通道，降維比例 r=8 → 8 神經元")
    print(f"  - 殘差塊 SE: 256 通道，降維比例 r=16 → 16 神經元")
    print(f"  - 總共 {10 + 1} 個 SE Block (10 個 ResBlock + 1 個輸入層)")