"""
向量存储模块
负责文档的向量化存储和检索
"""

import os
import pickle
from typing import List, Dict, Any

from .retriever import TFIDFRetriever


class VectorStore:
    """向量存储，使用TF-IDF进行文档向量化和检索"""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        初始化向量存储
        
        Args:
            persist_directory: 数据持久化目录
        """
        self.persist_directory = persist_directory
        self.retriever = TFIDFRetriever()
        self.collection_name = "weizhiyu_knowledge"
        self.documents = []
        self._ensure_directory()

    def _ensure_directory(self):
        """确保数据目录存在"""
        os.makedirs(self.persist_directory, exist_ok=True)

    def _get_db_path(self) -> str:
        """获取数据库文件路径"""
        return os.path.join(self.persist_directory, "tfidf_db.pkl")

    def create_collection(self, collection_name: str = "weizhiyu_knowledge"):
        """
        创建或加载集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否成功
        """
        self.collection_name = collection_name
        db_path = self._get_db_path()

        # 如果数据库文件存在，则加载
        if os.path.exists(db_path):
            self.load()

        return True

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        添加文档到向量存储
        
        Args:
            chunks: 文档chunks列表
            
        Returns:
            添加的文档数量
        """
        self.documents.extend(chunks)
        self.retriever.fit(self.documents)
        self.save()
        return len(chunks)

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        查询相关文档
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            
        Returns:
            相关文档列表
        """
        return self.retriever.query(query_text, n_results)

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            包含集合名称和文档数量的字典
        """
        return {
            "name": self.collection_name,
            "count": len(self.documents)
        }

    def save(self):
        """保存向量存储到文件"""
        db_path = self._get_db_path()
        data = {
            "documents": self.documents,
            "collection_name": self.collection_name
        }
        with open(db_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """
        从文件加载向量存储
        
        Returns:
            是否加载成功
        """
        db_path = self._get_db_path()
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
            self.documents = data.get("documents", [])
            self.collection_name = data.get("collection_name", "weizhiyu_knowledge")
            if self.documents:
                self.retriever.fit(self.documents)
            return True
        return False
