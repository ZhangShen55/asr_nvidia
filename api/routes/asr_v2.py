"""历史模块名兼容导入；应用不再注册 v2 路由。"""

from api.routes.asr_v3 import api_asr_v3, router

# 旧的 Python 导入名仅用于源码级迁移，不创建 /v2/seacraft_asr 路由。
api_asr_v2 = api_asr_v3

__all__ = ["router", "api_asr_v3", "api_asr_v2"]
