CONFIG = {
    'take_multiplier': 1.2,      #吃棋獎勵
    'kill_action': 15,      #和棋回合数
    'repetitionlimit': 3,   # 第三次相同局面判和
    'dirichlet': 0.075,       # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 100,        # 每次移动的模拟次数
    'c_puct': 3,             # u的权重
    'c_puct_for_fight': 1,
    'buffer_size': 100000,   # 经验池大小
    'paddle_model_path': 'current_policy.model',      # paddle模型路径
    'pytorch_model_path': 'current_policy.pth',   # pytorch模型路径
    'new_pytorch_model_path': 'new_current_policy.pth',
    'train_data_buffer_path': 'train_data_buffer.pth',   # 数据容器的路径
    'new_data_buffer_path': 'new_data_buffer.pth',
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 60,  #模型更新间隔时间
    'use_redis': False, # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
    'no_dark_mode': False
}