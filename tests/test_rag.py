"""
RAG系统测试
"""

import os
import sys
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.document_loader import DocumentLoader
from app.rag.text_splitter import TextSplitter
from app.rag.retriever import ChineseTokenizer, TFIDFRetriever
from app.rag.vector_store import VectorStore


class TestDocumentLoader(unittest.TestCase):
    """文档加载器测试"""

    def setUp(self):
        self.test_file = os.path.join(os.path.dirname(__file__), "..", "knowledge", "薇之雨完整知识手册.md")
        if os.path.exists(self.test_file):
            self.loader = DocumentLoader(self.test_file)
        else:
            self.skipTest("知识库文件不存在")

    def test_load(self):
        """测试加载文档"""
        content = self.loader.load()
        self.assertIsNotNone(content)
        self.assertGreater(len(content), 0)

    def test_parse_qa_pairs(self):
        """测试解析Q&A对"""
        self.loader.load()
        qa_pairs = self.loader.parse_qa_pairs()
        self.assertIsInstance(qa_pairs, list)
        self.assertGreater(len(qa_pairs), 0)
        
        # 检查第一个Q&A对的结构
        first_qa = qa_pairs[0]
        self.assertIn("question", first_qa)
        self.assertIn("content", first_qa)
        self.assertIn("metadata", first_qa)


class TestTextSplitter(unittest.TestCase):
    """文本分割器测试"""

    def setUp(self):
        self.splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

    def test_split_short_text(self):
        """测试分割短文本"""
        text = "这是一个测试文本。"
        chunks = self.splitter.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_split_long_text(self):
        """测试分割长文本"""
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。" * 10
        chunks = self.splitter.split_text(text)
        self.assertGreater(len(chunks), 1)


class TestChineseTokenizer(unittest.TestCase):
    """中文分词器测试"""

    def setUp(self):
        self.tokenizer = ChineseTokenizer()

    def test_tokenize(self):
        """测试分词"""
        text = "调肤原理是什么？"
        tokens = self.tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # 检查是否包含同义词扩展
        self.assertIn("调肤", tokens)


class TestTFIDFRetriever(unittest.TestCase):
    """TF-IDF检索器测试"""

    def setUp(self):
        self.retriever = TFIDFRetriever()
        self.documents = [
            {"text": "调肤原理是地基重建逻辑", "metadata": {"title": "调肤原理"}},
            {"text": "产品安全不含激素", "metadata": {"title": "产品安全"}},
            {"text": "多久见效因人而异", "metadata": {"title": "见效时间"}},
        ]
        self.retriever.fit(self.documents)

    def test_query(self):
        """测试查询"""
        results = self.retriever.query("调肤原理", n_results=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn("text", results[0])
        self.assertIn("score", results[0])


class TestVectorStore(unittest.TestCase):
    """向量存储测试"""

    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.store = VectorStore(self.test_dir)

    def tearDown(self):
        # 清理测试数据
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_add_and_query(self):
        """测试添加和查询"""
        documents = [
            {"text": "调肤原理测试", "metadata": {"title": "测试1"}},
            {"text": "产品安全测试", "metadata": {"title": "测试2"}},
        ]
        self.store.add_documents(documents)
        
        results = self.store.query("调肤", n_results=1)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
