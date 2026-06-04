"""
文本分割器模块
负责将长文本分割成适合向量化的chunks
"""

import re
from typing import List, Dict, Any


class TextSplitter:
    """文本分割器，支持按段落和句子分割文本"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        初始化文本分割器
        
        Args:
            chunk_size: 每个chunk的最大字符数
            chunk_overlap: chunk之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成chunks
        
        Args:
            text: 待分割的文本
            
        Returns:
            分割后的文本chunks列表
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        # 按段落分割
        paragraphs = re.split(r'\n\n+', text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前chunk加上新段落不超过限制，则合并
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 保存当前chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 如果段落本身超过限制，按句子进一步分割
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[。！？；])', para)
                    sub_chunk = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(sub_chunk) + len(sent) <= self.chunk_size:
                            sub_chunk += sent
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk.strip())
                            sub_chunk = sent
                    if sub_chunk.strip():
                        current_chunk = sub_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para

        # 添加最后一个chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def split_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分割Q&A对
        
        Args:
            qa_pairs: Q&A对列表
            
        Returns:
            分割后的chunks列表，每个chunk包含text和metadata
        """
        chunks = []

        for qa in qa_pairs:
            question = qa["question"]
            content = qa["content"].strip()
            chapter = qa.get("chapter", "")
            section = qa.get("section", "")
            metadata = qa.get("metadata", {})

            # 构建完整文本
            full_text = f"**{question}**\n\n{content}".strip()

            if len(full_text) <= self.chunk_size:
                chunks.append({
                    "text": full_text,
                    "metadata": {
                        **metadata,
                        "title": question,
                        "chapter": chapter,
                        "section": section,
                    }
                })
            else:
                # 分割长文本
                text_chunks = self.split_text(content)
                for i, chunk in enumerate(text_chunks):
                    prefix = f"**{question}**\n\n" if i == 0 else ""
                    chunk_text = f"{prefix}{chunk}".strip()
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "title": question,
                            "chapter": chapter,
                            "section": section,
                            "chunk_index": i,
                        }
                    })

        return chunks

    def split_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分割章节内容
        
        Args:
            sections: 章节列表
            
        Returns:
            分割后的chunks列表
        """
        chunks = []

        for section in sections:
            title = section["title"]
            content = section["content"]
            metadata = section["metadata"]

            if len(content) <= self.chunk_size:
                chunk_text = f"{title}\n\n{content}".strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {**metadata, "title": title}
                    })
            else:
                text_chunks = self.split_text(content)
                for i, chunk in enumerate(text_chunks):
                    chunk_text = f"{title}\n\n{chunk}".strip()
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                **metadata,
                                "title": title,
                                "chunk_index": i
                            }
                        })

        return chunks
