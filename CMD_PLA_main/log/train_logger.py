import os
import sys
# 获取当前脚本所在目录的父目录（项目根目录）
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import time
import json

from log.basic_logger import BasicLogger
from config_hg.config_dict import Config

def create_dir(dir_list):
    assert isinstance(dir_list, list), "dir_list must be a list"
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)  # 增加exist_ok避免重复创建报错

class TrainLogger(BasicLogger):
    def __init__(self, args, config_name, create=True):
        self.args = args
        self.config_name = config_name  # 保存配置文件名用于生成路径

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 生成实验标识（savetag）
        if args.get('mark') is None:
            savetag = f"{timestamp}_{args.get('model')}_repeat{args.get('repeat')}"
        else:
            savetag = f"{timestamp}_{args.get('model')}_repeat{args.get('repeat')}_{args.get('mark')}"

        # 1. 将save_dir转换为绝对路径（确保模型保存根目录是绝对路径）
        save_dir = args.get('save_dir')
        if save_dir is None:
            raise Exception('save_dir can not be None!')
        save_dir_abs = os.path.abspath(save_dir)  # 转换为绝对路径

        # 2. 生成实验相关目录的绝对路径
        train_save_dir = os.path.join(save_dir_abs, savetag)
        self.log_dir = os.path.abspath(os.path.join(train_save_dir, 'log', 'train'))
        self.model_dir = os.path.abspath(os.path.join(train_save_dir, 'model'))
        self.result_dir = os.path.abspath(os.path.join(train_save_dir, 'result'))

        if create:
            create_dir([self.log_dir, self.model_dir, self.result_dir])
            print(f"日志目录: {self.log_dir}")
            log_path = os.path.join(self.log_dir, 'Train.log')
            super().__init__(log_path)
            self.record_config()  # 去掉config参数，使用实例保存的config_name

    def record_config(self):
        # 3. 配置文件保存路径使用绝对路径
        config_save_path = os.path.abspath(os.path.join(self.log_dir, f'{self.config_name}.json'))
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.args, f, indent=2, ensure_ascii=False)  # 更规范的JSON格式化

    def get_log_dir(self):
        return self.log_dir if hasattr(self, 'log_dir') else None

    def get_model_dir(self):
        return self.model_dir if hasattr(self, 'model_dir') else None

    def get_result_dir(self):
        return self.result_dir if hasattr(self, 'result_dir') else None


if __name__ == "__main__":
    # 4. 配置文件路径使用绝对路径
    # 假设config目录在项目根目录下，生成绝对路径
    config_name = "TrainConfig"
    config_dir = os.path.abspath(os.path.join(parent_dir, 'config_hg'))  # 配置文件所在目录的绝对路径
    os.makedirs(config_dir, exist_ok=True)  # 确保config目录存在

    # 初始化配置（Config类内部会读取config/TrainConfig.json）
    config = Config(config_name)  # 传递配置文件名
    args = config.get_config()    # 获取配置参数

    # 确保save_dir在配置中存在且转换为绝对路径
    if 'save_dir' not in args:
        # 如果配置中没有save_dir，手动指定一个绝对路径
        args['save_dir'] = os.path.abspath(os.path.join(parent_dir, 'experiments'))

    # 初始化日志器
    logger = TrainLogger(args, config_name)  # 传递配置文件名
    model_path = logger.get_model_dir()
    print(f"模型保存目录: {model_path}")
