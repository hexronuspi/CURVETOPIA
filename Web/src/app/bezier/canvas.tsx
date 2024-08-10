"use client";

import { useEffect, useRef, useState } from 'react';
import * as fabric from 'fabric';

const DrawingCanvas = () => {
  const canvasRef = useRef<fabric.Canvas | null>(null);
  const [svgs, setSvgs] = useState<string[]>([]);

  useEffect(() => {
    if (canvasRef.current) return;

    const canvas = new fabric.Canvas('drawingCanvas', {
      width: 800,
      height: 600,
      backgroundColor: 'white',
    });

    canvas.isDrawingMode = true;
    canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
    canvas.freeDrawingBrush.color = '#000000'; 
    canvas.freeDrawingBrush.width = 5; 

    canvasRef.current = canvas;
    console.log('Fabric canvas initialized successfully');
  }, []);

  const captureCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      try {
        const svg = canvas.toSVG();
        setSvgs((prev) => [...prev, svg]);
        clearCanvas();  
      } catch (error) {
        console.error('Error capturing canvas:', error);
      }
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.clear();
    }
  };

  return (
    <div className="flex flex-col items-center bg-white p-6 rounded-lg shadow-lg">
      <canvas id="drawingCanvas" className="p-2 border-4 border-gray-800 mb-4 bg-white" />
      <button
        onClick={captureCanvas}
        className="px-6 py-3 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 transition duration-300 ease-in-out"
      >
        Save Drawing
      </button>
      <div className="mt-6">
        {svgs.map((svg, index) => (
          <div key={index} className="mb-4">
            <h2 className="text-xl font-semibold mb-2">Drawing {index + 1}:</h2>
            <div className="border border-gray-800 rounded-lg shadow-md max-w-full">
              <div
                dangerouslySetInnerHTML={{ __html: svg }}
                className="w-full h-full"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DrawingCanvas;
