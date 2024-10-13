# 使用官方 Python 镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app
RUN pip install --upgrade pip

# 复制 requirements.in 和 requirements.txt
COPY requirements.in .

# 安装 pip-tools
RUN pip install pip-tools

# 生成 requirements.txt
RUN pip-compile requirements.in
RUN ls -la
RUN ls
# 复制生成的 requirements.txt 文件
COPY requirements.txt .

# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 5000

# 启动 Flask 应用
CMD ["python", "app.py"]
