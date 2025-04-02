import React, { useEffect, useRef, useState } from 'react';
import {
  Holistic,
  HAND_CONNECTIONS
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
      refineFaceLandmarks: true,
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
    ctx.scale(-1, 1); //좌우반전
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

    if (results.faceLandmarks) {
      const l = results.faceLandmarks;
      drawLine([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], l); // 윗 입술
      drawLine([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291], l); // 아랫 입술
      drawLine([13, 82, 81, 80, 191, 78], l, 'red'); // 입술 중앙 선
      drawLine([33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33], l); // 왼쪽 눈눈
      drawLine([362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362], l); // 오른쪽 눈
      drawLine([70, 63, 105, 66, 107], l); // 왼쪽 눈썹
      drawLine([336, 296, 334, 293, 300], l); // 오른쪽 눈썹
    }

    const drawHandDetails = (landmarks, color) => {
      if (!landmarks) return;

      const fingers = [
        [0, 1, 2, 3, 4],     // 엄지
        [5, 6, 7, 8],       // 검지
        [9, 10, 11, 12],    // 중지
        [13, 14, 15, 16],   // 약지
        [17, 18, 19, 20]    // 소지
      ];

      const palmConnections = [
        [0, 5], [5, 9], [9, 13], [13, 17], [17, 0] // 손바닥
      ];

      fingers.forEach(finger => drawLine(finger, landmarks, color));
      palmConnections.forEach(([start, end]) => drawConnection(landmarks[start], landmarks[end], color));
      landmarks.forEach(p => drawPoint(p.x, p.y, color, 4));
    };

    drawHandDetails(results.leftHandLandmarks, 'purple'); // 왼손
    drawHandDetails(results.rightHandLandmarks, 'blue'); // 오른손

    if (results.poseLandmarks) {
      const l = results.poseLandmarks;
      const pointIndices = [11, 12, 13, 14, 15, 16];
      const connections = [
        [11, 13], [13, 15], // 왼쪽 어깨 -> 팔꿈치 -> 손목
        [12, 14], [14, 16], // 오른쪽 어깨 -> 팔꿈치 -> 손목
        [11, 12] // 양쪽 어깨 연결선
      ];
      pointIndices.forEach(i => drawPoint(l[i].x, l[i].y, 'green', 4)); // 각 관절 위치 초록색 점
      connections.forEach(([start, end]) => drawConnection(l[start], l[end], 'green')); // 각 관절 사이 초록색 선
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
