# 使用官方 Python 镜像
FROM python:3.8-alpine

# 设置工作目录
WORKDIR /app


# 复制 requirements.in
COPY requirements.txt .



# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 10000

# 启动 Flask 应用
CMD ["python", "app.py"]
