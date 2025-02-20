import os
import torch
import torch.nn as nn
import numpy as np
import ffmpeg
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2


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

# 批处理大小（batch_size）
BATCH_SIZE = 16


# ✅ 加载超分模型
from tools import load_sr_model
sr_model = load_sr_model("FSRCNN",4,device)


async def super_resolve_batch(frames):
    """ 使用 FSRCNN 进行超分辨率处理 """
    frames = frames.to(device)  
    with torch.no_grad():  # 关闭梯度计算，加速推理
        upsampled_frames = sr_model(frames)
    return (upsampled_frames * 255).clamp(0, 255).to(torch.uint8).cpu()

async def upscale_cbcr_batch(cb_batch, cr_batch, target_size):
    """ 执行 CbCr 插值（仍然是异步） """
    loop = asyncio.get_running_loop()
    cb_resized_list = await loop.run_in_executor(None, lambda: [cv2.resize(cb, target_size, interpolation=cv2.INTER_NEAREST) for cb in cb_batch])
    cr_resized_list = await loop.run_in_executor(None, lambda: [cv2.resize(cr, target_size, interpolation=cv2.INTER_NEAREST) for cr in cr_batch])
    return cb_resized_list, cr_resized_list



async def upscale_video_ffmpeg(input_path, output_path):
    """ 逐帧读取视频，进行批量超分辨率处理，并直接写入 FFmpeg """
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
        .output(output_path, pix_fmt="yuv420p", vcodec="libx264", r=fps)#, crf=23, preset="fast") 
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    frame_size = width * height * 3
    frame_count = 0
    frame_buffer = []
    cb_buffer = []
    cr_buffer = []

    while True:
        raw_frame = process_in.stdout.read(frame_size)
        if not raw_frame:
            break

        # 读取帧并存入批量缓冲区
        frame = np.frombuffer(raw_frame, np.uint8).reshape(height, width, 3)
        
        # ✅ 1. 转换为 YCbCr 格式（交换 Cb 和 Cr）
        frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        y_channel = frame_ycbcr[:, :, 0]
        cb_channel = frame_ycbcr[:, :, 2]  # 交换 Cb 和 Cr
        cr_channel = frame_ycbcr[:, :, 1]

        # 归一化 Y 通道并存入 buffer
        y_tensor = torch.tensor(y_channel, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
        frame_buffer.append(y_tensor)

        # 存入 CbCr 通道
        cb_buffer.append(cb_channel)
        cr_buffer.append(cr_channel)

        if len(frame_buffer) >= BATCH_SIZE:
            batch_tensor = torch.stack(frame_buffer).to(device)  # [B, 1, H, W]
            target_size = (width * 4, height * 4)

            # ✅ 串行执行（先 Y 超分，再 CbCr 插值）
            upsampled_y = await super_resolve_batch(batch_tensor)  # 1. Y 超分
            upsampled_y = upsampled_y.squeeze(1).cpu().numpy().astype(np.uint8)  # [B, 4H, 4W]

            cb_resized_list, cr_resized_list = await upscale_cbcr_batch(cb_buffer, cr_buffer, target_size)  # 2. CbCr 插值

            # ✅ 3. 重新组合 YCbCr 并转换回 RGB
            for i in range(BATCH_SIZE):
                ycrcb_upsampled = np.stack([upsampled_y[i], cr_resized_list[i],cb_resized_list[i]], axis=2)
                rgb_upsampled = cv2.cvtColor(ycrcb_upsampled, cv2.COLOR_YCrCb2RGB)
                process_out.stdin.write(rgb_upsampled.tobytes())  # 写入 FFmpeg

            # 清空缓冲区
            frame_buffer = []
            cb_buffer = []
            cr_buffer = []

        frame_count += 1
        if frame_count % 10 == 0:
            processing_status["progress"] = int((frame_count / total_frames) * 100)
            print(f"📈 处理进度: {processing_status['progress']}%")
        
        


     # ✅ 处理剩余的帧
    if frame_buffer:
        batch_tensor = torch.stack(frame_buffer).to(device)
        upsampled_y = await super_resolve_batch(batch_tensor)
        upsampled_y = upsampled_y.squeeze(1).cpu().numpy().astype(np.uint8)

        cb_resized_list, cr_resized_list = await upscale_cbcr_batch(cb_buffer, cr_buffer, (width * 4, height * 4))

        batch_rgb_frames = []
        for i in range(len(upsampled_y)):
            ycrcb_upsampled = np.stack([upsampled_y[i], cr_resized_list[i],cb_resized_list[i]], axis=2)
            rgb_upsampled = cv2.cvtColor(ycrcb_upsampled, cv2.COLOR_YCrCb2RGB)
            batch_rgb_frames.append(rgb_upsampled)

        process_out.stdin.write(np.array(batch_rgb_frames, dtype=np.uint8).tobytes())

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)