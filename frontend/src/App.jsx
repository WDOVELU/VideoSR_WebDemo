import React from "react";
import VideoComparisonPlayer from "./components/VideoComparisonPlayer";  // ✅ 确保路径正确

function App() {
    return (
        <div className="App">
            <h1>视频对比播放器</h1>
            <VideoComparisonPlayer />
        </div>
    );
}

export default App;