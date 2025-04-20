import { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';

interface Layer {
  neurons: number;
  activation?: string;
}

interface NeuralNetworkVizProps {
  layers: Layer[];
  width?: number;
  height?: number;
  className?: string;
}

export function NeuralNetworkViz({
  layers,
  width = 800,
  height = 400,
  className,
}: NeuralNetworkVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = width;
    canvas.height = height;

    // Animation variables
    let animationFrame: number;
    let time = 0;

    // Neural network parameters
    const neuronRadius = 8;
    const layerSpacing = width / (layers.length + 1);
    const maxNeurons = Math.max(...layers.map(l => l.neurons));
    const neuronSpacing = height / (maxNeurons + 1);

    // Draw functions
    const drawNeuron = (x: number, y: number, activation?: string) => {
      ctx.beginPath();
      ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
      ctx.fillStyle = activation ? '#60a5fa' : '#94a3b8';
      ctx.fill();
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 2;
      ctx.stroke();

      if (activation) {
        ctx.fillStyle = '#1e293b';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(activation, x, y + neuronRadius + 12);
      }
    };

    const drawConnection = (x1: number, y1: number, x2: number, y2: number) => {
      const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
      gradient.addColorStop(0, 'rgba(96, 165, 250, 0.2)');
      gradient.addColorStop(1, 'rgba(96, 165, 250, 0.1)');

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    const drawNetwork = () => {
      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Draw connections
      for (let i = 0; i < layers.length - 1; i++) {
        const currentLayer = layers[i];
        const nextLayer = layers[i + 1];
        const x1 = layerSpacing * (i + 1);
        const x2 = layerSpacing * (i + 2);

        for (let j = 0; j < currentLayer.neurons; j++) {
          const y1 = neuronSpacing * (j + 1);
          for (let k = 0; k < nextLayer.neurons; k++) {
            const y2 = neuronSpacing * (k + 1);
            drawConnection(x1, y1, x2, y2);
          }
        }
      }

      // Draw neurons
      layers.forEach((layer, i) => {
        const x = layerSpacing * (i + 1);
        for (let j = 0; j < layer.neurons; j++) {
          const y = neuronSpacing * (j + 1);
          drawNeuron(x, y, layer.activation);
        }
      });
    };

    const animate = () => {
      time += 0.01;
      drawNetwork();
      animationFrame = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, [layers, width, height]);

  return (
    <canvas
      ref={canvasRef}
      className={cn('rounded-lg bg-black/20 backdrop-blur-sm', className)}
    />
  );
} 