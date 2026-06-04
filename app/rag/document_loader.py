"""
文档加载器模块
负责加载和解析Markdown格式的知识库文档
"""

import re
from typing import List, Dict, Any
from pathlib import Path


class DocumentLoader:
    """文档加载器，支持解析Markdown格式的Q&A对和章节内容"""

    def __init__(self, file_path: str):
        """
        初始化文档加载器
        
        Args:
            file_path: 文档文件路径
        """
        self.file_path = file_path
        self.content = None
        self.qa_pairs = []

    def load(self) -> str:
        """
        加载文档内容
        
        Returns:
            文档内容字符串
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        return self.content

    def parse_qa_pairs(self) -> List[Dict[str, Any]]:
        """
        解析文档中的Q&A对
        
        Returns:
            Q&A对列表，每个元素包含question, content, chapter, section, metadata
        """
        if not self.content:
            self.load()

        qa_pairs = []
        lines = self.content.split('\n')

        current_qa = None
        current_chapter = ""
        current_section = ""

        for line in lines:
            # 解析一级标题（章节）
            if line.startswith('# ') and not line.startswith('## ') and not line.startswith('### '):
                if current_qa and current_qa["content"].strip():
                    qa_pairs.append(current_qa)
                    current_qa = None
                current_chapter = line.lstrip('#').strip()
                continue
            # 解析二级标题（小节）
            elif line.startswith('## '):
                if current_qa and current_qa["content"].strip():
                    qa_pairs.append(current_qa)
                    current_qa = None
                current_section = line.lstrip('#').strip()
                continue
            # 解析三级标题（问题）
            elif line.startswith('### '):
                heading_text = line.lstrip('#').strip()
                heading_clean = re.sub(r'^\d+\.\s*', '', heading_text)

                # 判断是否为问题格式
                is_question = bool(re.search(
                    r'[？?！!吗呢吧啊嘛呀]|'
                    r'是什么|怎么说|怎么找|怎么打|怎么让|怎么办|怎么做|'
                    r'如何|为什么|哪些|几个|能不能|会不会|有没有|'
                    r'原理|安全|效果|反弹|区别|差异|体系|流程|核心|关键|'
                    r'简介|介绍|背书|愿景|理念|实力|数据|'
                    r'案例|故事|经历|方法|策略|技巧|步骤|标准|要求|规范|'
                    r'选商|招商|融商|养商|裂商|循环|流水线|'
                    r'定海神针|差异化|批发|大客户|赚到|段公子|工厂|底层逻辑|'
                    r'农村|厨娘|CEO|七嫂|格林|市场',
                    heading_clean
                ))

                if is_question:
                    if current_qa and current_qa["content"].strip():
                        qa_pairs.append(current_qa)

                    current_qa = {
                        "question": heading_clean,
                        "content": "",
                        "chapter": current_chapter,
                        "section": current_section,
                        "metadata": {
                            "source": self.file_path,
                            "chapter": current_chapter,
                            "section": current_section,
                        }
                    }
                else:
                    if current_qa is not None:
                        current_qa["content"] += f"\n**{heading_clean}**\n"
                    continue

                continue

            # 收集问题的内容
            if current_qa is not None:
                current_qa["content"] += line + "\n"

        # 添加最后一个Q&A对
        if current_qa and current_qa["content"].strip():
            qa_pairs.append(current_qa)

        self.qa_pairs = qa_pairs
        return qa_pairs

    def parse_sections(self) -> List[Dict[str, Any]]:
        """
        解析文档中的章节内容
        
        Returns:
            章节列表，每个元素包含title, content, level, metadata
        """
        if not self.content:
            self.load()

        sections = []
        current_section = {
            "title": "薇之雨完整知识手册",
            "content": "",
            "level": 0,
            "metadata": {"source": self.file_path}
        }

        lines = self.content.split('\n')

        for line in lines:
            if line.startswith('# '):
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                current_section = {
                    "title": line[2:].strip(),
                    "content": "",
                    "level": 1,
                    "metadata": {
                        "source": self.file_path,
                        "section_type": "main"
                    }
                }
            elif line.startswith('## '):
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                current_section = {
                    "title": line[3:].strip(),
                    "content": "",
                    "level": 2,
                    "metadata": {
                        "source": self.file_path,
                        "section_type": "sub"
                    }
                }
            elif line.startswith('### '):
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                current_section = {
                    "title": line[4:].strip(),
                    "content": "",
                    "level": 3,
                    "metadata": {
                        "source": self.file_path,
                        "section_type": "detail"
                    }
                }
            else:
                current_section["content"] += line + "\n"

        if current_section["content"].strip():
            sections.append(current_section)

        return sections
