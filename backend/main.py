import os
import torch
import numpy as np
import av
import ffmpeg
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许 CORS，支持前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建 `temp` 目录存储视频
UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def bicubic_upsample(frame, scale_factor=4):
    """ 逐帧 Bicubic 插值 """
    frame = frame.unsqueeze(0)  # (1, C, H, W)
    upsampled_frame = torch.nn.functional.interpolate(
        frame, scale_factor=scale_factor, mode="bicubic", align_corners=False
    ).squeeze(0)  # (C, H', W')
    return (upsampled_frame * 255).clamp(0, 255).to(torch.uint8)

async def process_video_stream(file_path: str):
    """ 读取视频 → 超分 → FFmpeg 编码 → 流式传输 """
    input_container = av.open(file_path)

    # FFmpeg 进程：流式输出 Fragmented MP4
    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s="1280x720", r=30)
        .output("pipe:", format="mp4", movflags="frag_keyframe+empty_moov",
                vcodec="libx264", pix_fmt="yuv420p", r=30)
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )

    for frame in input_container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        upsampled_tensor = bicubic_upsample(img_tensor)
        upsampled_img = upsampled_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

        process.stdin.write(upsampled_img.tobytes())

        while True:
            chunk = process.stdout.read(1024 * 1024)  # 逐步读取 1MB
            if not chunk:
                break
            yield chunk  # ✅ 流式返回数据

    process.stdin.close()
    process.wait()

@app.post("/stream")
async def stream_video(video: UploadFile = File(...)):
    """ 处理上传视频，并流式返回超分流 """
    file_path = os.path.join(UPLOAD_FOLDER, video.filename)

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await video.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件写入失败: {str(e)}")

    return StreamingResponse(process_video_stream(file_path), media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)