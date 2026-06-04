FROM python:3.12-slim

LABEL maintainer="qisao-ai"
LABEL description="薇之雨 AI七嫂 智能问答系统"

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/status')" || exit 1

# 启动命令
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "wsgi:app"]
