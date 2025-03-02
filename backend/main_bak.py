import os
import torch
import numpy as np
import av
import ffmpeg
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建目录
UPLOAD_FOLDER = "videos"
OUTPUT_FOLDER = "videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 处理状态
processing_status = {"status": "idle", "progress": 0, "file": None}

# ✅ 选择设备（如果有 GPU，使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# 最大文件大小（500MB）
MAX_FILE_SIZE = 500 * 1024 * 1024


def bicubic_upsample(frame, scale_factor=4):
    """ 在 GPU 上进行 Bicubic 插值 """
    frame = frame.unsqueeze(0).to(device)  # ✅ 送入 GPU
    upsampled_frame = torch.nn.functional.interpolate(
        frame, scale_factor=scale_factor, mode="bicubic", align_corners=False
    ).squeeze(0)
    return (upsampled_frame * 255).clamp(0, 255).to(torch.uint8).cpu()  # ✅ 取回 CPU 方便处理


async def upscale_video_ffmpeg(input_path, output_path):
    """ 逐帧读取视频，进行 Bicubic 超分辨率，并直接写入 FFmpeg """
    print(f"📥 读取视频: {input_path}")
    processing_status["status"] = "processing"
    processing_status["progress"] = 0
    processing_status["file"] = output_path

    # 读取视频信息
    try:
        probe = ffmpeg.probe(input_path)
        video_info = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
        if not video_info:
            raise ValueError("No video stream found")
    except ffmpeg.Error as e:
        print("❌ FFmpeg probe error:", e)
        processing_status["status"] = "error"
        return

    width, height = int(video_info["width"]), int(video_info["height"])
    total_frames = int(video_info.get("nb_frames", 1000))  # 估算总帧数
    fps = eval(video_info["r_frame_rate"])

    process_in = (
        ffmpeg.input(input_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, pipe_stdin=True)
    )

    process_out = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width*4}x{height*4}", r=fps)
        .output(output_path, pix_fmt="yuv420p", vcodec="libx264", r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    frame_size = width * height * 3
    frame_count = 0

    while True:
        raw_frame = process_in.stdout.read(frame_size)
        if not raw_frame:
            break

        frame = np.frombuffer(raw_frame, np.uint8).reshape(height, width, 3)
        frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # ✅ 逐帧 GPU 处理
        upsampled_frame = bicubic_upsample(frame_tensor)
        upsampled_img = upsampled_frame.permute(1, 2, 0).numpy().astype(np.uint8)

        # ✅ 逐帧写入 FFmpeg
        process_out.stdin.write(upsampled_img.tobytes())

        frame_count += 1
        if frame_count % 10 == 0:
            processing_status["progress"] = int((frame_count / total_frames) * 100)
            print(f"📈 处理进度: {processing_status['progress']}%")

    process_in.wait()
    process_out.stdin.close()
    process_out.wait()

    if os.path.exists(output_path):
        processing_status["status"] = "completed"
        processing_status["progress"] = 100
        print("✅ 处理完成！")
    else:
        processing_status["status"] = "error"
        print("❌ 处理失败，文件未找到！")


@app.post("/upload")
async def upload_video(video: UploadFile = File(...), request: Request = None):
    """ 处理视频上传并异步超分 """
    if request and int(request.headers.get("content-length", 0)) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    file_path = os.path.join(UPLOAD_FOLDER, video.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"4k_{video.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    processing_status["status"] = "queued"
    processing_status["progress"] = 0

    # ✅ 启动异步任务处理
    asyncio.create_task(upscale_video_ffmpeg(file_path, output_path))

    return JSONResponse({"message": "Processing started", "output": f"4k_{video.filename}"})


@app.get("/status")
async def get_status():
    """ 获取处理状态 """
    return JSONResponse(processing_status)


@app.get("/videos/{filename}")
async def get_video(filename: str):
    """ 提供超分视频文件 """
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)