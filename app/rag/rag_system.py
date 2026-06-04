"""
RAG系统模块
整合文档加载、文本分割、向量存储和响应生成
"""

import os
import re
from typing import List, Dict, Any

from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .vector_store import VectorStore
from .response_generator import ResponseGenerator


class RAGSystem:
    """RAG系统，整合所有组件"""

    def __init__(self, knowledge_base_path: str, db_path: str = "./data/chroma_db", llm_config_path: str = None):
        """
        初始化RAG系统
        
        Args:
            knowledge_base_path: 知识库文件路径
            db_path: 向量数据库路径
            llm_config_path: LLM配置文件路径
        """
        self.knowledge_base_path = knowledge_base_path
        self.db_path = db_path
        self.loader = DocumentLoader(knowledge_base_path)
        self.splitter = TextSplitter(chunk_size=1600, chunk_overlap=200)
        self.vector_store = VectorStore(db_path)
        self.response_generator = ResponseGenerator(config_path=llm_config_path)

    def initialize(self):
        """
        初始化RAG系统，加载知识库并构建向量索引
        
        Returns:
            是否初始化成功
        """
        print("正在加载知识库...")
        self.loader.load()

        print("正在解析Q&A对...")
        qa_pairs = self.loader.parse_qa_pairs()
        print(f"解析到 {len(qa_pairs)} 个Q&A对")

        print("正在解析章节内容...")
        sections = self.loader.parse_sections()
        print(f"解析到 {len(sections)} 个章节")

        print("正在分割文本...")
        qa_chunks = self.splitter.split_qa_pairs(qa_pairs)
        section_chunks = self.splitter.split_sections(sections)

        # 去重
        chunks = []
        seen_text = set()
        for chunk in qa_chunks + section_chunks:
            text = re.sub(r'\s+', ' ', chunk["text"]).strip()
            if text and text not in seen_text:
                seen_text.add(text)
                chunks.append(chunk)

        print(f"生成 {len(chunks)} 个文本块")

        print("正在创建向量存储...")
        self.vector_store.create_collection()

        print("正在添加文档到向量数据库...")
        count = self.vector_store.add_documents(chunks)
        print(f"成功添加 {count} 个文档块")

        stats = self.vector_store.get_collection_stats()
        print(f"向量数据库统计: {stats}")

        return True

    def query(self, question: str, n_results: int = 8, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        查询RAG系统
        
        Args:
            question: 用户问题
            n_results: 返回结果数量
            history: 对话历史
            
        Returns:
            包含question, response, sources, num_results的字典
        """
        results = self.vector_store.query(question, n_results=n_results)
        response = self.response_generator.generate(question, results, history)

        return {
            "question": question,
            "response": response,
            "sources": results,
            "num_results": len(results)
        }

    def query_stream(self, question: str, n_results: int = 8, history: List[Dict[str, str]] = None):
        """
        流式查询RAG系统
        
        Args:
            question: 用户问题
            n_results: 返回结果数量
            history: 对话历史
            
        Yields:
            生成的回答文本片段
        """
        results = self.vector_store.query(question, n_results=n_results)
        yield from self.response_generator.generate_stream(question, results, history)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            统计信息字典
        """
        return self.vector_store.get_collection_stats()
