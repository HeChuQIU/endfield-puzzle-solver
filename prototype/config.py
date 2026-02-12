#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载模块
提供统一的配置文件读取和访问接口
"""

import toml
from pathlib import Path
from typing import Any, Dict, List, Optional


class Config:
    """配置管理类，支持点号分隔的路径访问"""

    def __init__(self, config_path: Path):
        """
        加载配置文件

        Args:
            config_path: TOML 配置文件路径
        """
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._raw: Dict = toml.load(f)

        self._config_path = config_path

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        支持点号分隔的路径访问

        Args:
            key_path: 点号分隔的键路径，如 'color_grouping.hue_merge_distance'
            default: 默认值，当路径不存在时返回

        Returns:
            配置值或默认值

        Examples:
            >>> config.get('color_grouping.hue_merge_distance')
            18
            >>> config.get('auto_scaling.search_range', [0.5, 2.0])
            [0.5, 2.0]
        """
        keys = key_path.split('.')
        value = self._raw

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_int(self, key_path: str, default: int = 0) -> int:
        """获取整数配置值"""
        return int(self.get(key_path, default))

    def get_float(self, key_path: str, default: float = 0.0) -> float:
        """获取浮点数配置值"""
        return float(self.get(key_path, default))

    def get_bool(self, key_path: str, default: bool = False) -> bool:
        """获取布尔配置值"""
        return bool(self.get(key_path, default))

    def get_list(self, key_path: str, default: Optional[List] = None) -> List:
        """获取列表配置值"""
        if default is None:
            default = []
        value = self.get(key_path, default)
        if not isinstance(value, list):
            return default
        return value

    def __repr__(self) -> str:
        return f"Config(path='{self._config_path}')"


def load_default_config() -> Config:
    """
    加载默认配置文件

    Returns:
        Config 对象
    """
    config_path = Path(__file__).parent / "image" / "config.toml"
    return Config(config_path)
