import React, { useEffect, useRef, useState } from "react";
import SimplePeer from "simple-peer";

const WebRTCPlayer = () => {
    const localVideoRef = useRef(null);
    const remoteVideoRef = useRef(null);
    const [peer, setPeer] = useState(null);

    useEffect(() => {
        const startWebRTC = async () => {
            try {
                console.log("🔵 获取本地视频流...");
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });

                if (localVideoRef.current) {
                    localVideoRef.current.srcObject = stream;
                }

                console.log("✅ 本地视频流获取成功，初始化 WebRTC 连接...");
                const newPeer = new SimplePeer({
                    initiator: true,
                    trickle: false,
                    stream,
                });

                newPeer.on("signal", async (data) => {
                    try {
                        console.log("📡 发送 WebRTC Offer 到服务器...");
                        const response = await fetch("http://localhost:8000/offer", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(data),
                        });

                        if (!response.ok) {
                            throw new Error("服务器返回错误");
                        }

                        const answer = await response.json();
                        console.log("✅ 收到 WebRTC Answer，应用到 peer...");
                        newPeer.signal(answer);
                    } catch (error) {
                        console.error("❌ WebRTC 信令错误:", error);
                    }
                });

                newPeer.on("stream", (remoteStream) => {
                    console.log("🎥 收到远程 WebRTC 视频流...");
                    if (remoteVideoRef.current) {
                        remoteVideoRef.current.srcObject = remoteStream;
                    }
                });

                newPeer.on("connect", () => {
                    console.log("✅ WebRTC 连接成功！");
                });

                newPeer.on("error", (err) => {
                    console.error("❌ WebRTC 连接错误:", err);
                });

                setPeer(newPeer);
            } catch (error) {
                console.error("❌ WebRTC 连接失败:", error);
            }
        };

        startWebRTC();

        return () => {
            console.log("🛑 清理 WebRTC 连接...");
            if (peer) {
                peer.destroy();
            }
        };
    }, []);

    return (
        <div>
            <h2>WebRTC 实时超分辨率</h2>
            <div style={{ display: "flex", gap: "20px" }}>
                <video ref={localVideoRef} autoPlay muted style={{ width: "50%" }} />
                <video ref={remoteVideoRef} autoPlay style={{ width: "50%" }} />
            </div>
        </div>
    );
};

export default WebRTCPlayer;