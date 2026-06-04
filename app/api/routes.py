"""
API路由模块
"""

import os
import json
from flask import Blueprint, request, jsonify, Response, stream_with_context

# 创建蓝图
api_bp = Blueprint('api', __name__)

# 全局RAG系统实例
rag_system = None


def init_rag(system):
    """初始化RAG系统"""
    global rag_system
    rag_system = system


@api_bp.route("/query", methods=["POST"])
def query():
    """查询接口"""
    global rag_system
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        history = data.get("history", [])

        if not question:
            return jsonify({"status": "error", "error": "问题不能为空"})

        if not rag_system:
            return jsonify({"status": "error", "error": "系统正在初始化，请稍候"})

        result = rag_system.query(question, n_results=8, history=history)

        # 将Markdown转换为HTML
        import markdown2
        markdown_text = result.get("response", "")
        html = markdown2.markdown(
            markdown_text,
            extras=["fenced-code-blocks", "tables", "strike", "task_list", "header-ids", "link-protocols"]
        )

        return jsonify({"status": "ok", "html": html})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@api_bp.route("/query_stream", methods=["POST"])
def query_stream():
    """流式查询接口"""
    global rag_system
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        history = data.get("history", [])

        if not question:
            return jsonify({"status": "error", "error": "问题不能为空"})

        if not rag_system:
            return jsonify({"status": "error", "error": "系统正在初始化，请稍候"})

        def event_stream():
            """SSE事件流"""
            import markdown2
            yield f"data: {json.dumps({'type': 'start'}, ensure_ascii=False)}\n\n"
            markdown_parts = []
            try:
                for chunk in rag_system.query_stream(question, n_results=8, history=history):
                    if not chunk:
                        continue
                    markdown_parts.append(chunk)
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
                
                # 生成最终HTML
                markdown_text = "".join(markdown_parts)
                html = markdown2.markdown(
                    markdown_text,
                    extras=["fenced-code-blocks", "tables", "strike", "task_list", "header-ids", "link-protocols"]
                )
                yield f"data: {json.dumps({'type': 'done', 'html': html}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

        return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@api_bp.route("/status")
def status():
    """系统状态接口"""
    global rag_system
    if rag_system:
        stats = rag_system.get_stats()
        return jsonify({"status": "ok", "count": stats["count"]})
    return jsonify({"status": "initializing"})
