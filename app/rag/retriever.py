"""
检索器模块
包含中文分词器和TF-IDF检索器
"""

import re
import math
from typing import List, Dict, Any
from collections import Counter


class ChineseTokenizer:
    """中文分词器，支持同义词扩展"""

    def __init__(self):
        # 停用词表
        self.stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
            '吗', '那', '被', '从', '把', '些', '之', '而', '但', '与',
            '对', '为', '以', '所', '如', '过', '来', '能', '可', '这个',
            '那个', '什么', '怎么', '为什么', '可以', '因为', '所以', '如果',
            '虽然', '但是', '或者', '还是', '不是', '没有', '已经', '正在',
            '将要', '可能', '应该', '必须', '需要', '可以', '能够', '愿意',
        ])

        # 同义词映射
        self.synonyms = {
            '调肤': ['调肤', '调理', '修复', '改善皮肤'],
            '安全': ['安全', '激素', '铅汞', '违禁', '检测'],
            '反弹': ['反弹', '依赖', '停用', '反黑'],
            '效果': ['效果', '见效', '改善', '结果'],
            '价格': ['价格', '多少钱', '费用', '收费', '成本'],
            '产品': ['产品', '护肤品', '套盒', '方案'],
            '过敏': ['过敏', '敏感', '红肿', '刺激'],
            '斑': ['斑', '色斑', '色素', '黑色素', '黄褐斑', '雀斑'],
            '痘': ['痘', '痘痘', '痤疮', '粉刺', '闭口'],
            '门店': ['门店', '店', '拓客', '引流', '客源'],
            '批发': ['批发', '代理', '加盟', '合作', '进货'],
        }

    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 待分词的文本
            
        Returns:
            分词结果列表
        """
        text = text.lower()
        text = re.sub(r'[^\u4e00-\u9fa5a-z0-9]', ' ', text)

        tokens = []

        # 同义词扩展
        for key, syns in self.synonyms.items():
            for syn in syns:
                if syn in text:
                    tokens.append(key)
                    break

        # 中文字符n-gram
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]+', text)
        for chars in chinese_chars:
            for i in range(len(chars)):
                tokens.append(chars[i])
                if i + 1 < len(chars):
                    tokens.append(chars[i:i+2])
                if i + 2 < len(chars):
                    tokens.append(chars[i:i+3])

        # 英文单词
        english_words = re.findall(r'[a-z]+', text)
        tokens.extend(english_words)

        # 数字
        numbers = re.findall(r'[0-9]+', text)
        tokens.extend(numbers)

        # 过滤停用词
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 0]

        return tokens


class TFIDFRetriever:
    """基于TF-IDF的检索器"""

    def __init__(self):
        self.tokenizer = ChineseTokenizer()
        self.documents = []
        self.doc_tokens = []
        self.idf = {}
        self.tfidf_vectors = []

    def fit(self, documents: List[Dict[str, Any]]):
        """
        训练TF-IDF模型
        
        Args:
            documents: 文档列表，每个文档包含text字段
        """
        self.documents = documents

        # 对所有文档进行分词
        self.doc_tokens = []
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc["text"])
            self.doc_tokens.append(tokens)

        # 计算文档频率
        df = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        # 计算IDF
        n_docs = len(documents)
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((n_docs + 1) / (freq + 1)) + 1

        # 计算TF-IDF向量
        self.tfidf_vectors = []
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            total = len(tokens)

            vector = {}
            for token, count in tf.items():
                tfidf = (count / total) * self.idf.get(token, 0)
                vector[token] = tfidf

            self.tfidf_vectors.append(vector)

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        查询相关文档
        
        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            
        Returns:
            相关文档列表，包含text, metadata, score
        """
        query_tokens = self.tokenizer.tokenize(query_text)

        # 计算查询向量
        query_tf = Counter(query_tokens)
        query_total = len(query_tokens)

        query_vector = {}
        for token, count in query_tf.items():
            tfidf = (count / query_total) * self.idf.get(token, 0)
            query_vector[token] = tfidf

        # 计算相似度
        scores = []
        for i, doc_vector in enumerate(self.tfidf_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((score, i))

        # 排序并返回top结果
        scores.sort(reverse=True)

        results = []
        for score, idx in scores[:n_results]:
            results.append({
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "score": score
            })

        return results

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度
        """
        common_keys = set(vec1.keys()) & set(vec2.keys())

        if not common_keys:
            return 0.0

        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)

        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
