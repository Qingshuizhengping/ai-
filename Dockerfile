FROM python:3.12-slim

LABEL maintainer="qisao-ai"
LABEL description="薇之雨 AI七嫂 智能问答系统"

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量默认值
ENV PORT=5000

# 暴露端口
EXPOSE ${PORT}

# 启动命令
CMD python run.py
