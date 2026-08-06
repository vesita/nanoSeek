"""
YAML 配置加载器：替代旧的 "poor man's configurator"（exec Python 配置文件）。

用法（从项目根目录）：
    python train.py config/train_shakespeare_char.yaml --n_layer=4 --max_iters=1000

- 第一个不带 '=' 的参数：YAML 配置文件，覆盖 train.py 里的全局变量
- 其余 --key=value 参数：命令行覆盖，值用 literal_eval 解析（支持 int/float/bool/str）
- 类型检查：配置里的值必须和 train.py 全局变量的类型一致（和旧 configurator 同样的约束）
"""
import sys
from ast import literal_eval

import yaml


def load_config(g):
    """把 YAML 配置 + 命令行覆盖应用到调用者的 globals()（即 g）。"""
    # 1. 找出配置文件（第一个不带 '=' 的参数）
    config_file = None
    for arg in sys.argv[1:]:
        if '=' not in arg:
            assert not arg.startswith('--'), f"配置文件不能以 '--' 开头: {arg}"
            config_file = arg

    # 2. 加载并应用 YAML 配置
    if config_file is not None:
        with open(config_file) as f:
            data = yaml.safe_load(f) or {}
        print(f"已加载配置 {config_file}")
        for k, v in data.items():
            if k in g:
                # 类型必须一致，避免 yaml 里的 '2' 把 int 全局变量悄悄变成 str 之类的问题
                if type(v) is not type(g[k]):
                    raise TypeError(
                        f"配置键 '{k}': yaml 里是 {type(v).__name__} = {v!r}, "
                        f"但 train.py 期望 {type(g[k]).__name__}"
                    )
                g[k] = v
            else:
                # 未知键：仍然注入（和旧 exec 方案行为一致），并提示一下
                print(f"note: 配置键 '{k}' 不是 train.py 的全局变量，仍注入为全局变量")
                g[k] = v

    # 3. 应用命令行 --key=value 覆盖
    for arg in sys.argv[1:]:
        if '=' in arg:
            assert arg.startswith('--'), f"命令行覆盖必须以 '--' 开头: {arg}"
            key, val = arg.split('=', 1)
            key = key[2:]
            if key not in g:
                raise ValueError(f"Unknown config key: {key}")
            if val.lower() in ('true', 'false'):
                # 'true'/'false' 是 YAML 常用写法，但 literal_eval 只认 Python 的 True/False
                attempt = val.lower() == 'true'
            else:
                try:
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # 解析不了就当成字符串
                    attempt = val
            if type(attempt) is not type(g[key]):
                raise TypeError(
                    f"命令行覆盖 '{key}={val}' 类型是 {type(attempt).__name__}, "
                    f"但期望 {type(g[key]).__name__}"
                )
            print(f"Overriding: {key} = {attempt}")
            g[key] = attempt
