import React, { useEffect, useRef, useState } from 'react';
import {
  Holistic
} from '@mediapipe/holistic';
import { Camera as MediapipeCamera } from '@mediapipe/camera_utils';
import './CSS/Camera.css';

const Camera = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [stream, setStream] = useState(null);

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  const toggleCamera = async () => {
    if (isCameraOn) {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      setStream(null);
      videoRef.current.srcObject = null;
      clearCanvas();
    } else {
      try {
        const newStream = await navigator.mediaDevices.getUserMedia({ video: true });
        setStream(newStream);
        videoRef.current.srcObject = newStream;
        setupHolisticTracking();
      } catch (error) {
        console.error('Error accessing the camera', error);
      }
    }
    setIsCameraOn((prevState) => !prevState);
  };

  const setupHolisticTracking = () => {
    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: false, // 얼굴 인식 끔
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults(onResults);

    const camera = new MediapipeCamera(videoRef.current, {
      onFrame: async () => {
        await holistic.send({ image: videoRef.current });
      },
      width: 640,
      height: 480,
    });
    camera.start();
  };

  const onResults = (results) => {
    if (!canvasRef.current || !videoRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    ctx.save();
    ctx.scale(-1, 1); // 좌우반전
    ctx.drawImage(videoRef.current, -canvasRef.current.width, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.restore();

    drawLandmarks(ctx, results);
  };

  const drawLandmarks = (ctx, results) => {
    const drawPoint = (x, y, color = 'red', radius = 3) => {
      ctx.beginPath();
      ctx.arc((1 - x) * canvasRef.current.width, y * canvasRef.current.height, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    };

    const drawConnection = (p1, p2, color = 'blue') => {
      ctx.beginPath();
      ctx.moveTo((1 - p1.x) * canvasRef.current.width, p1.y * canvasRef.current.height);
      ctx.lineTo((1 - p2.x) * canvasRef.current.width, p2.y * canvasRef.current.height);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    };

    const drawLine = (indices, landmarks, color = 'orange') => {
      ctx.beginPath();
      indices.forEach((idx, i) => {
        const x = (1 - landmarks[idx].x) * canvasRef.current.width;
        const y = landmarks[idx].y * canvasRef.current.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    };

    const drawHandDetails = (landmarks, color) => {
      if (!landmarks) return;

      const fingers = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]
      ];

      const palmConnections = [
        [0, 5], [5, 9], [9, 13], [13, 17], [17, 0]
      ];

      fingers.forEach(finger => drawLine(finger, landmarks, color));
      palmConnections.forEach(([start, end]) => drawConnection(landmarks[start], landmarks[end], color));
      landmarks.forEach(p => drawPoint(p.x, p.y, color, 4));
    };

    drawHandDetails(results.leftHandLandmarks, 'purple');
    drawHandDetails(results.rightHandLandmarks, 'blue');

    if (results.poseLandmarks) {
      const l = results.poseLandmarks;
      const pointIndices = [11, 12, 13, 14, 15, 16];
      const connections = [
        [11, 13], [13, 15],
        [12, 14], [14, 16],
        [11, 12]
      ];
      pointIndices.forEach(i => drawPoint(l[i].x, l[i].y, 'green', 4));
      connections.forEach(([start, end]) => drawConnection(l[start], l[end], 'green'));
    }
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext('2d');
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
