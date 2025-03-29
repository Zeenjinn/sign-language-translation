import React, { useEffect, useRef, useState } from 'react';  
import './Camera.css'; // CSS 파일 import  

const Camera = () => {  
  const videoRef = useRef(null);  
  const [isCameraOn, setIsCameraOn] = useState(false);  
  const [stream, setStream] = useState(null);  

  useEffect(() => {  
    return () => {  
      // Cleanup function to stop the video stream  
      if (stream) {  
        const tracks = stream.getTracks();  
        tracks.forEach(track => track.stop());  
      }  
    };  
  }, [stream]);  

  const toggleCamera = async () => {  
    if (isCameraOn) {  
      // Stop the stream  
      if (stream) {  
        const tracks = stream.getTracks();  
        tracks.forEach(track => track.stop());  
      }  
      videoRef.current.srcObject = null;  
    } else {  
      // Start the stream  
      try {  
        const newStream = await navigator.mediaDevices.getUserMedia({ video: true });  
        setStream(newStream);  
        videoRef.current.srcObject = newStream;  
      } catch (error) {  
        console.error("Error accessing the camera", error);  
      }  
    }  
    setIsCameraOn(prevState => !prevState);  
  };  

  return (  
    <div className="camera-container">  
      <h1 className="title">Camera Stream</h1>  
      <video ref={videoRef} autoPlay playsInline className="video" style={{ display: isCameraOn ? 'block' : 'none' }} />  
      <button onClick={toggleCamera} className="button">  
        {isCameraOn ? 'Stop Camera' : 'Start Camera'}  
      </button>  
    </div>  
  );  
};  

export default Camera;