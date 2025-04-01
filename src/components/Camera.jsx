import React, { useEffect, useRef, useState } from 'react';  
import { Hands } from "@mediapipe/hands";
import { Camera as MediapipeCamera } from "@mediapipe/camera_utils";
import './CSS/Camera.css';

const Camera = () => {  
  const videoRef = useRef(null);  
  const canvasRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);  
  const [stream, setStream] = useState(null);
  
  useEffect(() => {  
    return () => {  
      if (stream) {  
        stream.getTracks().forEach(track => track.stop());  
      }  
    };  
  }, [stream]);  

  const toggleCamera = async () => {  
    if (isCameraOn) {  
      if (stream) {  
        stream.getTracks().forEach(track => track.stop());  
      }  
      setStream(null);
      videoRef.current.srcObject = null;  
      clearCanvas();
    } else {  
      try {  
        const newStream = await navigator.mediaDevices.getUserMedia({ video: true });  
        setStream(newStream);  
        videoRef.current.srcObject = newStream;
        setupHandTracking();
      } catch (error) {  
        console.error("Error accessing the camera", error);  
      }  
    }  
    setIsCameraOn(prevState => !prevState);  
  };  

  const setupHandTracking = () => {
    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    hands.setOptions({
      maxNumHands: 2, // 손 감지 개수
      modelComplexity: 1, // 모델 복잡도 설정
      minDetectionConfidence: 0.5, // 최소 감지 신뢰도 설정
      minTrackingConfidence: 0.5, // 최소 추적 신뢰도 설정
    });
    hands.onResults(onResults);

    const camera = new MediapipeCamera(videoRef.current, {
      onFrame: async () => {
        await hands.send({ image: videoRef.current });
      },
      width: 640,
      height: 480,
    });
    camera.start();
  };

  const onResults = (results) => {
    if (!canvasRef.current || !videoRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    ctx.save();
    ctx.scale(-1, 1); // 좌우반전
    ctx.drawImage(videoRef.current, -canvasRef.current.width, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.restore();
    
    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        drawLandmarks(ctx, landmarks);
      }
    }
  };

  // 선
  const drawLandmarks = (ctx, landmarks) => {
    ctx.strokeStyle = "blue"; // 선 색상상
    ctx.lineWidth = 2; // 선 두께께

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // 엄지
      [5, 6], [6, 7], [7, 8], // 검지
      [9, 10], [10, 11], [11, 12], // 중지
      [13, 14], [14, 15], [15, 16], // 약지
      [17, 18], [18, 19], [19, 20], // 소지
      [0, 5], [5, 9], [9, 13], [13, 17], [0, 17] // 손바닥
    ];

    connections.forEach(([start, end]) => {
      ctx.beginPath();
      ctx.moveTo((1 - landmarks[start].x) * canvasRef.current.width, landmarks[start].y * canvasRef.current.height);
      ctx.lineTo((1 - landmarks[end].x) * canvasRef.current.width, landmarks[end].y * canvasRef.current.height);
      ctx.stroke();
    });

    // 점
    landmarks.forEach(point => {
      ctx.beginPath();
      ctx.arc((1 - point.x) * canvasRef.current.width, point.y * canvasRef.current.height, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red"; // 점 색상상
      ctx.fill();
    });
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  return (  
    <div className="camera-container">  
      <h1 className="title">Camera Stream</h1>  
      <canvas 
        ref={canvasRef} 
        className="video-canvas" 
        width={640} 
        height={480} 
        style={{ display: isCameraOn ? 'block' : 'none' }}
      ></canvas>
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        className="video" 
        style={{ display: 'none' }} 
      />
      <button onClick={toggleCamera} className="button">  
        {isCameraOn ? 'Stop Camera' : 'Start Camera'}  
      </button>  
    </div>  
  );  
};  

export default Camera;
