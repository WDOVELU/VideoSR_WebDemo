import React, { useState, useEffect, useRef } from "react";

const StreamingPlayer = () => {
    const [uploading, setUploading] = useState(false);
    const videoRef = useRef(null);
    const mediaSource = useRef(new MediaSource());

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setUploading(true);
        const formData = new FormData();
        formData.append("video", file);

        try {
            console.log("📡 上传视频到后端...");
            const response = await fetch("http://localhost:8000/stream", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("上传失败");

            console.log("✅ 开始流式播放...");
            playStreamingVideo(response);
        } catch (error) {
            console.error("❌ 视频上传失败:", error);
        }
        setUploading(false);
    };

    const playStreamingVideo = (response) => {
        const video = videoRef.current;
        video.src = URL.createObjectURL(mediaSource.current);

        mediaSource.current.onsourceopen = async () => {
            const sourceBuffer = mediaSource.current.addSourceBuffer('video/mp4; codecs="avc1.640029, mp4a.40.2"');

            const reader = response.body.getReader();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                sourceBuffer.appendBuffer(value);
            }
            mediaSource.current.endOfStream();
        };
    };

    return (
        <div>
            <h2>流式超分辨率播放器</h2>
            <input type="file" accept="video/mp4" onChange={handleFileUpload} />
            {uploading && <p>⏳ 正在上传视频...</p>}
            <video ref={videoRef} controls autoPlay style={{ width: "100%" }} />
        </div>
    );
};

export default StreamingPlayer;