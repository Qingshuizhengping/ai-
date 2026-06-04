"""
AI七嫂 - 薇之雨品牌智能问答系统
主程序入口
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import create_app, init_rag_system
from app.api.routes import init_rag
from app.config import Config


def main():
    """主函数"""
    print("=" * 60)
    print("  薇之雨 · AI七嫂智能问答系统")
    print("=" * 60)
    
    # 创建Flask应用
    app = create_app()
    
    # 初始化RAG系统
    print("\n正在初始化知识库...")
    rag = init_rag_system()
    
    # 将RAG系统传递给API模块
    init_rag(rag)
    
    print("\n" + "=" * 60)
    print("系统初始化完成！")
    print(f"访问地址: http://{Config.HOST}:{Config.PORT}")
    print("=" * 60)
    
    # 启动服务器
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )


if __name__ == "__main__":
    main()
