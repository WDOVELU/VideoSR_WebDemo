video-comparison-player/
│── backend/                # FastAPI 后端
│   ├── videos/             # 存储上传和处理后的视频
│   ├── main.py             # FastAPI 服务器
│── frontend/               # React 前端
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoComparisonPlayer.jsx  # 主要组件
│   │   ├── App.jsx        # React 入口
│   │   ├── main.jsx       # React 渲染入口
│   ├── index.html         # HTML 模板
│   ├── package.json       # 依赖管理
│── README.md              # 项目说明
│── requirements.txt       # Python 依赖
│── run.sh                 # 启动前后端的脚本