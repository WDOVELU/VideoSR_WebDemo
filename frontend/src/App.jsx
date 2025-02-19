import React from "react";
import StreamingPlayer from "./components/StreamingPlayer";  // ✅ 确保路径正确

function App() {
    return (
        <div className="App">
            <h1>视频对比播放器</h1>
            <StreamingPlayer />
        </div>
    );
}

export default App;