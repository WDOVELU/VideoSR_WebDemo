import os
import torch
import numpy as np
import av
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

# WebRTC 连接池
pcs = set()

def bicubic_upsample(frame_tensor, scale_factor=4):
    """ 使用 PyTorch 进行 Bicubic 插值 """
    
    frame_tensor = frame_tensor.unsqueeze(0)  # (1, C, H, W)
    upsampled_frame = torch.nn.functional.interpolate(
        frame_tensor, scale_factor=scale_factor, mode="bicubic", align_corners=False
    ).squeeze(0)  # (C, H', W')

    return (upsampled_frame * 255).byte()

class VideoTransformTrack(MediaStreamTrack):
    """ WebRTC 处理视频流 """
    kind = "video"
    def __init__(self, track):
        super().__init__()
        self.track = track
        
    async def recv(self):
        """ 处理 WebRTC 视频帧并进行超分辨率 """
        frame = await self.track.recv()
        img = frame.to_ndarray(format="rgb24")

        # 转换为 PyTorch Tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_upsampled = bicubic_upsample(img_tensor)

        # 转换回 numpy 并构造帧
        upsampled_img = img_upsampled.permute(1, 2, 0).numpy().astype(np.uint8)
        new_frame = av.VideoFrame.from_ndarray(upsampled_img, format="rgb24")
        new_frame.pts = frame.pts

        return new_frame

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ WebRTC 连接管理 """
    yield  # 应用运行时暂停执行，直到应用关闭

    print("🛑 FastAPI 服务器正在关闭，释放 WebRTC 连接...")
    tasks = []
    while pcs:
        pc = pcs.pop()
        try:
            tasks.append(pc.close())  # ✅ 并行关闭所有连接
        except Exception as e:
            print(f"⚠️ 关闭 WebRTC 连接时出错: {e}")

    await asyncio.gather(*tasks)  # ✅ 并行等待所有连接关闭
    print("✅ 所有 WebRTC 连接已关闭")

app = FastAPI(lifespan=lifespan)


@app.post("/offer")
async def webrtc_offer(offer: dict):
    """ 处理 WebRTC 连接请求 """
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print(f"📡 收到 WebRTC 视频流: {track.kind}")
        if track.kind == "video":
            video_track = VideoTransformTrack(track)
            pc.addTrack(video_track)

    @pc.on("iceconnectionstatechange")
    async def on_ice_state_change():
        print(f"🛑 ICE 状态变化: {pc.iceConnectionState}")
        if pc.iceConnectionState == "disconnected":
            await close_webrtc_connection(pc)

    try:
        offer_desc = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        await pc.setRemoteDescription(offer_desc)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        print("✅ WebRTC 连接成功，返回 Answer")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    
    except Exception as e:
        print(f"❌ WebRTC Offer 处理失败: {e}")
        return {"error": str(e)}

async def close_webrtc_connection(pc):
    """ 关闭 WebRTC 连接 """
    print("🔴 关闭 WebRTC 连接")
    await pc.close()
    pcs.discard(pc)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)