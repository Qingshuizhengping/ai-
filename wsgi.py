"""
WSGI入口文件
用于Gunicorn等WSGI服务器
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import create_app, init_rag_system
from app.api.routes import init_rag

# 创建Flask应用
app = create_app()

# 初始化RAG系统
print("正在初始化知识库...")
rag = init_rag_system()

# 将RAG系统传递给API模块
init_rag(rag)

print("知识库初始化完成！")
