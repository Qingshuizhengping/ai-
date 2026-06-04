"""
配置文件
"""

import os
from pathlib import Path


class Config:
    """应用配置类"""
    
    # 基础路径
    BASE_DIR = Path(__file__).parent.parent
    
    # 知识库路径
    KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge", "薇之雨完整知识手册.md")
    
    # 数据库路径
    DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
    
    # LLM配置路径
    LLM_CONFIG_PATH = os.path.join(BASE_DIR, "llm_config.json")
    
    # Flask配置
    SECRET_KEY = os.environ.get("SECRET_KEY", "ai-qisao-secret-key-2024")
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    # 服务器配置
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))
    
    # 静态文件路径
    STATIC_FOLDER = os.path.join(BASE_DIR, "app", "static")
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, "app", "templates")
    
    # 头像路径
    AVATAR_PATH = os.path.join(BASE_DIR, "app", "static", "images", "avatar.png")
    
    @classmethod
    def init_app(cls, app):
        """初始化应用配置"""
        # 确保数据目录存在
        os.makedirs(cls.DB_PATH, exist_ok=True)
