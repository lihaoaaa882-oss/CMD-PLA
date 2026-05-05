import json
import os


class Config(object):
    def __init__(self, config_name, train=True):
        # 获取项目根目录（根据实际目录结构调整）
        # 假设 config_hg 目录的父目录是项目根目录
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # 配置文件的绝对路径
        self.config_path = os.path.join(self.root_dir, "config_hg", f"{config_name}.json")

        if train:
            self.mode = 'train'
        else:
            self.mode = 'test'

        # 读取配置文件（使用绝对路径）
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
            if self.mode == 'train':
                self.train_config = config_data.get('train', {})
            else:
                self.test_config = config_data.get('test', {})

    def get_mode(self):
        return self.mode

    def get_config(self):
        if self.mode == 'train':
            return self.train_config
        elif self.mode == 'test':
            return self.test_config

    def show_config(self):
        print('=' * 50)
        if self.mode == 'train':
            for key, value in self.train_config.items():
                print(f'{key}: {value}')
        elif self.mode == 'test':
            for key, value in self.test_config.items():
                print(f'{key}: {value}')
        print('=' * 50)