"""模型插件包：导入本包即完成所有模型注册。"""

from vtrain.models.registry import available_models, build_model, register

# 触发各模型的 @register（顺序无关；放在函数导入之后避免循环导入）
from vtrain.models import baseline, lgbm, logreg  # noqa: F401  # isort: skip

__all__ = ["available_models", "build_model", "register"]
