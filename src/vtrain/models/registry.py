"""模型注册表：名字 -> 工厂函数。加一个模型只需新增一个文件并 @register。"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator

from vtrain.config import AppConfig, FeatureSpec

ModelFactory = Callable[[AppConfig, FeatureSpec], BaseEstimator]

_REGISTRY: dict[str, ModelFactory] = {}


def register(name: str) -> Callable[[ModelFactory], ModelFactory]:
    def decorator(factory: ModelFactory) -> ModelFactory:
        if name in _REGISTRY:
            raise ValueError(f"模型已注册，重复注册: {name}")
        _REGISTRY[name] = factory
        return factory

    return decorator


def build_model(name: str, cfg: AppConfig, spec: FeatureSpec) -> BaseEstimator:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的模型: {name}；可用={sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg, spec)


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
