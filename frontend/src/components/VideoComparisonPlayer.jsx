import React, { useState, useEffect, useRef } from "react";

const VideoComparisonPlayer = () => {
    const [originalVideoURL, setOriginalVideoURL] = useState(null);
    const [processedVideoURL, setProcessedVideoURL] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [progress, setProgress] = useState(0);

    const originalVideoRef = useRef(null);
    const processedVideoRef = useRef(null);

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setUploading(true);
        setProcessing(true);
        setProgress(0);

        const formData = new FormData();
        formData.append("video", file);

        try {
            console.log("📡 上传视频到后端...");
            const response = await fetch("http://localhost:8000/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.message || "上传失败");

            console.log("✅ 视频上传成功，等待处理...");
            setOriginalVideoURL(URL.createObjectURL(file));

            // **启动轮询状态**
            pollProcessingStatus(data.output);
        } catch (error) {
            console.error("❌ 视频上传失败:", error);
            setUploading(false);
            setProcessing(false);
        }
    };

    // **轮询服务器状态**
    const pollProcessingStatus = (outputFilename) => {
        const interval = setInterval(async () => {
            const response = await fetch("http://localhost:8000/status");
            const statusData = await response.json();

            console.log("⏳ 处理状态:", statusData);
            setProgress(statusData.progress); // 更新前端进度条

            if (statusData.status === "completed") {
                clearInterval(interval);
                setProcessing(false);
                setProcessedVideoURL(`http://localhost:8000/videos/${outputFilename}`);
                console.log("✅ 处理完成，播放视频！");
            }

            if (statusData.status === "error") {
                clearInterval(interval);
                setProcessing(false);
                console.error("❌ 处理失败！");
            }
        }, 1000); // **每 5 秒检查一次状态**
    };

    // **同步播放**
    useEffect(() => {
        const originalVideo = originalVideoRef.current;
        const processedVideo = processedVideoRef.current;

        if (!originalVideo || !processedVideo) return;

        // **同步播放进度**
        const syncPlayback = () => {
            if (Math.abs(originalVideo.currentTime - processedVideo.currentTime) > 0.1) {
                processedVideo.currentTime = originalVideo.currentTime;
            }
        };

        // **播放 & 暂停同步**
        const syncPlayPause = (event) => {
            if (event.type === "play") {
                processedVideo.play();
            } else {
                processedVideo.pause();
            }
        };

        originalVideo.addEventListener("timeupdate", syncPlayback);
        originalVideo.addEventListener("play", syncPlayPause);
        originalVideo.addEventListener("pause", syncPlayPause);

        return () => {
            originalVideo.removeEventListener("timeupdate", syncPlayback);
            originalVideo.removeEventListener("play", syncPlayPause);
            originalVideo.removeEventListener("pause", syncPlayPause);
        };
    }, [originalVideoURL, processedVideoURL]);

    return (
        <div>
            <h2>视频对比播放器</h2>
            <input type="file" accept="video/mp4" onChange={handleFileUpload} />
            
            {uploading && <p>✅ 视频上传成功，等待处理...</p>}
            {processing && <p>⏳ 处理进度: {progress}%</p>}

            <div style={{ display: "flex", gap: "20px", marginTop: "10px" }}>
                {originalVideoURL && (
                    <video ref={originalVideoRef} src={originalVideoURL} controls width="50%" />
                )}
                {processedVideoURL && (
                    <video ref={processedVideoRef} src={processedVideoURL} controls width="50%" />
                )}
            </div>
        </div>
    );
};

export default VideoComparisonPlayer;