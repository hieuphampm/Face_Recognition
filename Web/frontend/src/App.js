import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <h2>Real-time Face Recognition</h2>
      <img
        src="http://localhost:5000/video_feed"
        alt="Webcam Feed"
        width="640"
        height="480"
        style={{ border: '2px solid #333', borderRadius: '10px' }}
      />
    </div>
  );
}

export default App;
