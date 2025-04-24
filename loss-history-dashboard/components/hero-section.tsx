"use client"

import { useEffect, useRef } from 'react'
import { Card, CardContent } from "@/components/ui/card"
import { Cpu, BarChart2, Brain, Gauge } from "lucide-react"

export function HeroSection() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions to match parent container
    const resizeCanvas = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.offsetWidth;
        canvas.height = parent.offsetHeight;
      }
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Particle animation
    const particles: Particle[] = [];
    const particleCount = 100;
    
    class Particle {
      x: number;
      y: number;
      radius: number;
      color: string;
      speedX: number;
      speedY: number;
      connectionRadius: number;
      
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.radius = Math.random() * 2 + 1;
        this.color = `rgba(66, 153, 225, ${Math.random() * 0.5 + 0.25})`;
        this.speedX = Math.random() * 2 - 1;
        this.speedY = Math.random() * 2 - 1;
        this.connectionRadius = 150;
      }
      
      draw() {
        if (!ctx) return;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
      }
      
      update() {
        if (this.x > canvas.width || this.x < 0) {
          this.speedX = -this.speedX;
        }
        if (this.y > canvas.height || this.y < 0) {
          this.speedY = -this.speedY;
        }
        
        this.x += this.speedX;
        this.y += this.speedY;
        
        this.draw();
      }
    }
    
    function connect() {
      for (let i = 0; i < particles.length; i++) {
        for (let j = i; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < particles[i].connectionRadius) {
            if (!ctx) return;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(66, 153, 225, ${1 - distance / particles[i].connectionRadius})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
    }
    
    // Create particles
    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }
    
    // Animation loop
    function animate() {
      requestAnimationFrame(animate);
      if (!ctx) return;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      for (const particle of particles) {
        particle.update();
      }
      
      connect();
    }
    
    animate();
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);
  
  return (
    <div className="relative overflow-hidden rounded-xl border shadow-xl bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-950 dark:to-indigo-900 mb-8">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ zIndex: 1 }} />
      
      <div className="relative z-10 px-6 py-12 md:py-16 md:px-12">
        <div className="mx-auto max-w-6xl">
          <h1 className="text-4xl md:text-5xl font-extrabold text-center mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 drop-shadow-sm">
            Loss History AI Pipeline
          </h1>
          
          <p className="text-center text-gray-700 dark:text-gray-300 text-lg md:text-xl max-w-3xl mx-auto mb-10">
            Advanced machine learning platform that automates insurance claim processing with exceptional accuracy and efficiency
          </p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-8">
            <MetricCard 
              title="AI Models" 
              value="5" 
              description="Trained models" 
              icon={<Brain className="w-8 h-8 text-blue-500" />} 
              iconBgColor="bg-blue-100 dark:bg-blue-900"
            />
            
            <MetricCard 
              title="Precision" 
              value="94.2%" 
              description="Prediction accuracy" 
              icon={<Gauge className="w-8 h-8 text-purple-500" />} 
              iconBgColor="bg-purple-100 dark:bg-purple-900"
            />
            
            <MetricCard 
              title="Processing" 
              value="43%" 
              description="Faster than manual" 
              icon={<Cpu className="w-8 h-8 text-green-500" />} 
              iconBgColor="bg-green-100 dark:bg-green-900"
            />
            
            <MetricCard 
              title="Features" 
              value="42" 
              description="ML features used" 
              icon={<BarChart2 className="w-8 h-8 text-amber-500" />}
              iconBgColor="bg-amber-100 dark:bg-amber-900" 
            />
          </div>
        </div>
      </div>
    </div>
  )
}

function MetricCard({ 
  title, 
  value, 
  description, 
  icon,
  iconBgColor
}: { 
  title: string; 
  value: string; 
  description: string; 
  icon: React.ReactNode;
  iconBgColor: string;
}) {
  return (
    <Card className="bg-white/80 dark:bg-gray-800/80 backdrop-blur border-0 shadow-lg overflow-hidden transform transition-transform hover:scale-105">
      <CardContent className="p-6">
        <div className="flex items-start">
          <div className={`p-3 rounded-full mr-4 ${iconBgColor}`}>
            {icon}
          </div>
          <div>
            <div className="font-bold text-lg">{title}</div>
            <div className="text-3xl font-extrabold text-gray-800 dark:text-gray-100">{value}</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">{description}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 