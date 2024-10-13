# 使用官方 Python 镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 升级 pip
RUN pip install --upgrade pip

# 复制 requirements.in
COPY requirements.in .

# 安装 pip-tools
RUN pip install pip-tools

# 生成 requirements.txt
RUN pip-compile requirements.in

# 打印当前目录内容以确认 requirements.txt 是否存在
RUN ls -la

# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 10000

# 启动 Flask 应用
CMD ["python", "app.py"]
