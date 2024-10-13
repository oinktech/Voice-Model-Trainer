# 基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露 Flask 端口
EXPOSE 10000

# 运行 Flask 应用
CMD ["python", "app.py"]
