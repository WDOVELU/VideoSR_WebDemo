#!/bin/bash

# 启动后端（后台运行）
echo "🚀 启动后端（FastAPI）..."
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &  # 后端运行日志写入 backend.log
BACKEND_PID=$!  # 记录后端进程 ID
cd ..

# 等待后端启动
sleep 2  

# 启动前端
echo "🚀 启动前端（React）..."
cd frontend
npm run dev > frontend.log 2>&1 &  # 前端运行日志写入 frontend.log
FRONTEND_PID=$!  # 记录前端进程 ID
cd ..

# 输出进程信息
echo "✅ 后端运行中（PID: $BACKEND_PID） | 日志: backend/backend.log"
echo "✅ 前端运行中（PID: $FRONTEND_PID） | 日志: frontend/frontend.log"

# 让脚本保持运行，防止进程被杀死
wait