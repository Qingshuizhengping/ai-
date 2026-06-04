"""
Flask应用主模块
"""

import os
import sys
from flask import Flask, render_template, send_file

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.rag import RAGSystem
from app.api.routes import api_bp, init_rag


def create_app(config_class=Config):
    """
    创建Flask应用
    
    Args:
        config_class: 配置类
        
    Returns:
        Flask应用实例
    """
    app = Flask(
        __name__,
        static_folder=config_class.STATIC_FOLDER,
        template_folder=config_class.TEMPLATE_FOLDER
    )
    
    # 加载配置
    app.config.from_object(config_class)
    config_class.init_app(app)
    
    # 注册蓝图
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # 注册路由
    register_routes(app)
    
    return app


def register_routes(app):
    """注册路由"""
    
    @app.route("/")
    def index():
        """主页"""
        return render_template("index.html")
    
    @app.route("/avatar.png")
    def avatar():
        """头像"""
        avatar_path = Config.AVATAR_PATH
        if os.path.exists(avatar_path):
            return send_file(avatar_path, mimetype="image/png")
        # 返回默认头像
        default_avatar = os.path.join(Config.STATIC_FOLDER, "images", "default_avatar.png")
        if os.path.exists(default_avatar):
            return send_file(default_avatar, mimetype="image/png")
        return "", 404


def init_rag_system():
    """初始化RAG系统"""
    # 检查知识库文件是否存在
    if not os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        print(f"错误: 知识库文件不存在: {Config.KNOWLEDGE_BASE_PATH}")
        sys.exit(1)
    
    # 创建RAG系统
    rag = RAGSystem(
        knowledge_base_path=Config.KNOWLEDGE_BASE_PATH,
        db_path=Config.DB_PATH,
        llm_config_path=Config.LLM_CONFIG_PATH
    )
    
    # 检查是否需要初始化
    db_file = os.path.join(Config.DB_PATH, "tfidf_db.pkl")
    if not os.path.exists(db_file):
        print("首次运行，正在构建向量数据库...")
        rag.initialize()
    else:
        print("检测到已有数据库，正在加载...")
        rag.vector_store.create_collection()
    
    stats = rag.get_stats()
    print(f"知识库就绪，包含 {stats['count']} 个文档块")
    
    return rag
