import React, { useEffect, useRef, useState } from 'react';
import { Holistic } from '@mediapipe/holistic';
import { Camera as MediapipeCamera } from '@mediapipe/camera_utils';
import './CSS/Camera.css';

const Camera = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [stream, setStream] = useState(null);
  const [result, setResult] = useState('');
  const sequence = useRef([]);
  const lastPredictionTime = useRef(Date.now());

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
      setResult(''); 
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
      refineFaceLandmarks: false,
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
    camera.fps = 10;
    camera.start();
  };

  const onResults = async (results) => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(videoRef.current, -canvasRef.current.width, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.restore();

    drawLandmarks(ctx, results);

    const keypoints = [];

    if (results.poseLandmarks) {
      [11, 13, 15, 12, 14, 16].forEach(i => {
        const lm = results.poseLandmarks[i];
        keypoints.push(lm.x, lm.y, lm.z);
      });
    } else {
      keypoints.push(...Array(18).fill(0));
    }

    const processHand = (landmarks) => {
      landmarks.forEach(lm => keypoints.push(lm.x, lm.y, lm.z));
    };

    if (results.leftHandLandmarks && results.rightHandLandmarks) {
      processHand(results.leftHandLandmarks);
      processHand(results.rightHandLandmarks);
    } else if (results.leftHandLandmarks || results.rightHandLandmarks) {
      processHand(results.leftHandLandmarks || results.rightHandLandmarks);
      keypoints.push(...Array(63).fill(0));
    } else {
      keypoints.push(...Array(126).fill(0));
    }

    if (keypoints.length === 144) {
      sequence.current.push(keypoints);
    }

    if (sequence.current.length >= 30) {
      const now = Date.now();
      if (now - lastPredictionTime.current > 1000) {
        const slicedSequence = sequence.current.slice(-30);

        console.log('📦 전송되는 시퀀스:', slicedSequence);

        try {
          const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequence: slicedSequence }),
          });

          const data = await response.json();
          console.log('🎯 예측 결과:', data);
          
          const prediction = data.prediction || data.result;

          if (prediction && data.confidence > 0.8) {
            setResult(`${prediction} (${(data.confidence * 100).toFixed(1)}%)`);
          } else {
            setResult('인식 실패 또는 신뢰도 낮음');
          }
        } catch (err) {
          console.error('❌ 백엔드 요청 실패:', err);
        }

        lastPredictionTime.current = now;
        // 시퀀스 초기화 없이 계속 유지
      }
    }
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
      <h1 className="title">Sign Language Translator</h1>
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
      {result && (
        <div className="result-box">
          <h2>🔍 예측 결과: {result}</h2>
        </div>
      )}
    </div>
  );
};

export default Camera;
