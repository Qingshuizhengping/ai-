"""
响应生成器模块
负责根据检索结果生成符合七嫂风格的回答
"""

import os
import re
import json
from typing import List, Dict, Any


class WebSearcher:
    """网络搜索器，用于在知识库匹配度低时进行网络搜索"""

    def __init__(self):
        self.enabled = True

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        执行网络搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            搜索结果列表
        """
        if not self.enabled:
            return []

        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, region="cn-zh", max_results=max_results))

            for r in search_results:
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })

            return results

        except Exception as e:
            print(f"网络搜索失败: {e}")
            return []


class ResponseGenerator:
    """响应生成器，支持LLM调用和本地回退"""

    # 七嫂的系统提示词
    SYSTEM_PROMPT = """你是薇之雨品牌创始人七嫂。你的说话风格：

1. 说话直接干脆，不绕弯子，不喜欢废话。开口就是重点，没有"我觉得""可能""大概"这类模糊词。
2. 对店家说话直接指出问题，不会为了照顾情绪而说软话。说话不客气但也不伤人。
3. 说话方式接地气，不端不装。会讲自己的真实经历和感受。把自己放在和店家平等的位置，不摆架子。
4. 习惯用事实和案例说话，不讲大道理，只讲谁做了什么、拿到了什么结果。
5. 该硬的时候硬，该软的时候有分寸。提到信任的门店、跟着干的员工、支持她的家人时会感性，但不煽情，一句就收。
6. 整体感觉像大姐在跟自家人说话。不哄着你，但你知道她为你着急。不跟你客气，但你知道她靠得住。语言简洁有力。
7. 不要喊人"姐"，直接说话。
8. 回答时优先严格依据知识库内容。涉及卡项权益、价格、流程、产品功效、案例数据时，必须按知识库原文事实回答，不要自由发挥。只有知识库材料不完整且问题需要经营建议时，才可以结合美业经营逻辑做少量补充。
9. 用自然流畅的口语回答，不要像念稿子。
10. 如果回答用到了网络搜索结果，要说明信息来源。
11. 不要在回答末尾加"注"、"备注"、"说明"等补充说明，回答完就结束，不要画蛇添足。"""

    def __init__(self, config_path: str = None):
        """
        初始化响应生成器
        
        Args:
            config_path: LLM配置文件路径
        """
        self.client = None
        self.model = "mimo-v2.5"
        self.max_tokens = 2048
        self.temperature = 0.7
        self.extra_body = {}
        self.web_searcher = WebSearcher()
        
        # 本地回退模式的开场白和结束语
        self._fallback_openings = [
            "这个问题问到点子上了。",
            "说白了，",
            "记住啊，",
            "我跟你讲，",
            "说真的，",
        ]
        self._fallback_closings = [
            "听明白了吗？听话照做就对了。",
            "懂了吗？有什么不明白的随时问我。",
            "记住，美业就是修心，修好了自己的心，客户自然就来了。",
            "知道怎么做了吧？赶紧行动起来。",
            "就这么干，别犹豫。",
        ]

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            self._load_from_env()

    def _load_config(self, config_path: str):
        """
        加载LLM配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            from openai import OpenAI

            self.client = OpenAI(
                api_key=config.get("api_key", ""),
                base_url=config.get("base_url", "https://api.openai.com/v1"),
            )
            self.model = config.get("model", "mimo")
            self.max_tokens = config.get("max_tokens", 1024)
            self.temperature = config.get("temperature", 0.7)
            self.extra_body = config.get("extra_body", {})

            print(f"LLM已配置: model={self.model}, base_url={config.get('base_url', 'default')}")
        except Exception as e:
            print(f"LLM配置加载失败，将使用本地回退模式: {e}")
            self.client = None

    def _load_from_env(self):
        """
        从环境变量加载LLM配置
        """
        import sys
        api_key = os.environ.get('OPENAI_API_KEY', '')
        base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        model = os.environ.get('OPENAI_MODEL', 'deepseek-ai/DeepSeek-V4-Flash')

        # 调试日志 - 输出到stderr确保Railway能捕获
        sys.stderr.write(f"[DEBUG] OPENAI_API_KEY 是否设置: {bool(api_key)} (长度={len(api_key)})\n")
        sys.stderr.write(f"[DEBUG] OPENAI_BASE_URL: {base_url}\n")
        sys.stderr.write(f"[DEBUG] OPENAI_MODEL: {model}\n")
        sys.stderr.flush()

        if api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                self.model = model
                self.max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS', '1024'))
                self.temperature = float(os.environ.get('OPENAI_TEMPERATURE', '0.7'))

                msg = f"LLM已从环境变量配置: model={self.model}, base_url={base_url}"
                print(msg)
                sys.stderr.write(f"[DEBUG] {msg}\n")
                sys.stderr.flush()
            except Exception as e:
                err_msg = f"从环境变量加载LLM配置失败: {e}"
                print(err_msg)
                sys.stderr.write(f"[DEBUG] {err_msg}\n")
                sys.stderr.flush()
                self.client = None
        else:
            msg = "未找到OPENAI_API_KEY环境变量，将使用本地回退模式"
            print(msg)
            sys.stderr.write(f"[DEBUG] {msg}\n")
            sys.stderr.flush()

    def generate(self, question: str, results: List[Dict[str, Any]], history: List[Dict[str, str]] = None) -> str:
        """
        生成回答
        
        Args:
            question: 用户问题
            results: 检索结果
            history: 对话历史
            
        Returns:
            生成的回答文本
        """
        # 去重和清理上下文
        context_parts = []
        seen = set()
        for r in results[:6]:
            fp = self._fingerprint(r["text"])
            if fp not in seen:
                seen.add(fp)
                clean = self._clean_markdown(r["text"])
                if clean and not any(self._is_mostly_duplicate(p, clean) for p in context_parts):
                    context_parts.append(clean)

        kb_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        # 判断是否需要网络搜索
        best_score = results[0]["score"] if results else 0
        need_web_search = best_score < 0.05 or not context_parts

        web_context = ""
        web_sources = []
        if need_web_search:
            print(f"知识库匹配度较低({best_score:.4f})，触发网络搜索...")
            web_results = self.web_searcher.search(question, max_results=3)
            if web_results:
                web_parts = []
                for wr in web_results:
                    web_parts.append(f"**{wr['title']}**\n{wr['body']}")
                    web_sources.append(wr["href"])
                web_context = "\n\n---\n\n".join(web_parts)

        # 如果没有上下文，返回默认回答
        if not kb_context and not web_context:
            return self._no_answer()

        # 根据是否有LLM选择生成方式
        if self.client:
            return self._generate_with_llm(question, kb_context, web_context, web_sources, history)
        else:
            full_context = kb_context
            if web_context:
                full_context += "\n\n---\n\n" + web_context if full_context else web_context
            return self._generate_fallback(question, full_context)

    def generate_stream(self, question: str, results: List[Dict[str, Any]], history: List[Dict[str, str]] = None):
        """
        流式生成回答
        
        Args:
            question: 用户问题
            results: 检索结果
            history: 对话历史
            
        Yields:
            生成的回答文本片段
        """
        # 去重和清理上下文
        context_parts = []
        seen = set()
        for r in results[:6]:
            fp = self._fingerprint(r["text"])
            if fp not in seen:
                seen.add(fp)
                clean = self._clean_markdown(r["text"])
                if clean and not any(self._is_mostly_duplicate(p, clean) for p in context_parts):
                    context_parts.append(clean)

        kb_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        # 判断是否需要网络搜索
        best_score = results[0]["score"] if results else 0
        need_web_search = best_score < 0.05 or not context_parts

        web_context = ""
        web_sources = []
        if need_web_search:
            print(f"知识库匹配度较低({best_score:.4f})，触发网络搜索...")
            web_results = self.web_searcher.search(question, max_results=3)
            if web_results:
                web_parts = []
                for wr in web_results:
                    web_parts.append(f"**{wr['title']}**\n{wr['body']}")
                    web_sources.append(wr["href"])
                web_context = "\n\n---\n\n".join(web_parts)

        # 如果没有上下文，返回默认回答
        if not kb_context and not web_context:
            yield self._no_answer()
            return

        # 根据是否有LLM选择生成方式
        if self.client:
            yield from self._generate_with_llm_stream(question, kb_context, web_context, web_sources, history)
            return

        full_context = kb_context
        if web_context:
            full_context += "\n\n---\n\n" + web_context if full_context else web_context
        yield self._generate_fallback(question, full_context)

    def _generate_with_llm(self, question: str, kb_context: str, web_context: str, web_sources: List[str], history: List[Dict[str, str]] = None) -> str:
        """
        使用LLM生成回答
        
        Args:
            question: 用户问题
            kb_context: 知识库上下文
            web_context: 网络搜索上下文
            web_sources: 网络来源
            history: 对话历史
            
        Returns:
            生成的回答
        """
        try:
            context_sections = []

            if kb_context:
                context_sections.append(f"【知识库内容】\n\n{kb_context}")

            if web_context:
                context_sections.append(f"【网络搜索结果】\n\n{web_context}")

            history_text = self._format_history(history)
            if history_text:
                context_sections.insert(0, f"【最近对话上下文】\n\n{history_text}")

            full_context = "\n\n---\n\n".join(context_sections)

            # 根据上下文来源设置提示
            source_hint = ""
            if kb_context and web_context:
                source_hint = "知识库和网络搜索都有相关内容，请把它们当作参考材料，优先吸收知识库里的品牌表达和事实，结合你的判断回答。"
            elif web_context and not kb_context:
                source_hint = "知识库中没有找到相关内容，以下为网络搜索结果，请把网络搜索结果当作参考材料，结合你的判断回答，并在回答末尾注明信息来源于网络搜索。"
            elif kb_context and not web_context:
                source_hint = "以下为知识库内容，请严格按知识库事实回答。涉及卡项权益、价格、流程、产品功效、案例数据时，不要自由发挥，不要漏掉表格里的关键项。"

            user_message = f"""{full_context}

---

用户问题：{question}

{source_hint}

请用七嫂的风格快速回答用户的问题。要求：
1. 不要展示思考过程，不要铺垫，直接给结论和做法
2. 回答控制在重点范围内，能三句话说清就不要写十句
3. 涉及知识库已有事实时必须按知识库回答，不要自己改数字、改权益、改流程
4. 如果需要步骤，最多给3到5条
5. 如果用到了网络搜索结果，在回答末尾标注来源链接
6. 不要在回答末尾加"注"、"备注"、"说明"等补充说明，回答完就结束"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=self.extra_body,
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            print(f"LLM调用失败，使用本地回退: {e}")
            fallback_context = "\n\n---\n\n".join(
                part for part in [kb_context, web_context] if part
            )
            return self._generate_fallback(question, fallback_context)

    def _generate_with_llm_stream(self, question: str, kb_context: str, web_context: str, web_sources: List[str], history: List[Dict[str, str]] = None):
        """
        使用LLM流式生成回答
        
        Args:
            question: 用户问题
            kb_context: 知识库上下文
            web_context: 网络搜索上下文
            web_sources: 网络来源
            history: 对话历史
            
        Yields:
            生成的回答文本片段
        """
        try:
            context_sections = []

            if kb_context:
                context_sections.append(f"【知识库内容】\n\n{kb_context}")

            if web_context:
                context_sections.append(f"【网络搜索结果】\n\n{web_context}")

            history_text = self._format_history(history)
            if history_text:
                context_sections.insert(0, f"【最近对话上下文】\n\n{history_text}")

            full_context = "\n\n---\n\n".join(context_sections)

            # 根据上下文来源设置提示
            source_hint = ""
            if kb_context and web_context:
                source_hint = "知识库和网络搜索都有相关内容，请把它们当作参考材料，优先吸收知识库里的品牌表达和事实，结合你的判断回答。"
            elif web_context and not kb_context:
                source_hint = "知识库中没有找到相关内容，以下为网络搜索结果，请把网络搜索结果当作参考材料，结合你的判断回答，并在回答末尾注明信息来源于网络搜索。"
            elif kb_context and not web_context:
                source_hint = "以下为知识库内容，请严格按知识库事实回答。涉及卡项权益、价格、流程、产品功效、案例数据时，不要自由发挥，不要漏掉表格里的关键项。"

            user_message = f"""{full_context}

---

用户问题：{question}

{source_hint}

请用七嫂的风格快速回答用户的问题。要求：
1. 不要展示思考过程，不要铺垫，直接给结论和做法
2. 回答控制在重点范围内，能三句话说清就不要写十句
3. 涉及知识库已有事实时必须按知识库回答，不要自己改数字、改权益、改流程
4. 如果需要步骤，最多给3到5条
5. 如果用到了网络搜索结果，在回答末尾标注来源链接
6. 不要在回答末尾加"注"、"备注"、"说明"等补充说明，回答完就结束"""

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                extra_body=self.extra_body,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content

        except Exception as e:
            print(f"LLM流式调用失败，使用本地回退: {e}")
            fallback_context = "\n\n---\n\n".join(
                part for part in [kb_context, web_context] if part
            )
            yield self._generate_fallback(question, fallback_context)

    def _generate_fallback(self, question: str, context: str) -> str:
        """
        本地回退模式生成回答
        
        Args:
            question: 用户问题
            context: 上下文内容
            
        Returns:
            格式化的回答
        """
        import random
        opening = random.choice(self._fallback_openings)
        closing = random.choice(self._fallback_closings)
        return f"{opening}\n\n{context}\n\n---\n\n> 💡 {closing}"

    def _format_history(self, history: List[Dict[str, str]] = None) -> str:
        """
        格式化对话历史
        
        Args:
            history: 对话历史列表
            
        Returns:
            格式化的历史文本
        """
        if not history:
            return ""

        lines = []
        for item in history[-8:]:
            role = item.get("role", "")
            content = re.sub(r'\s+', ' ', item.get("content", "")).strip()
            if not content:
                continue
            if len(content) > 300:
                content = content[:300] + "..."
            name = "用户" if role == "user" else "AI七嫂"
            lines.append(f"{name}: {content}")

        return "\n".join(lines)

    def _clean_markdown(self, text: str) -> str:
        """
        清理Markdown文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        text = re.sub(r'^\*\*.*?\*\*\s*\n*', '', text, count=1)
        text = re.sub(r'^#{1,4}\s+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _fingerprint(self, text: str) -> str:
        """
        生成文本指纹用于去重
        
        Args:
            text: 输入文本
            
        Returns:
            文本指纹
        """
        clean = re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text.lower())
        return clean[:80]

    def _is_mostly_duplicate(self, text1: str, text2: str) -> bool:
        """
        判断两段文本是否大部分重复
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            是否重复
        """
        c1 = set(re.findall(r'[\u4e00-\u9fa5]{2,4}', text1))
        c2 = set(re.findall(r'[\u4e00-\u9fa5]{2,4}', text2))
        if not c1 or not c2:
            return False
        overlap = len(c1 & c2) / min(len(c1), len(c2))
        return overlap > 0.6

    def _no_answer(self) -> str:
        """
        无法回答时的默认响应
        
        Returns:
            默认回答文本
        """
        return """这个问题我暂时没找到答案。

但是记住啊，美业的核心就那几个：

1. **产品要好**
2. **服务要到位**
3. **人要用心**

你具体想问的是哪方面？我再帮你找找。"""
