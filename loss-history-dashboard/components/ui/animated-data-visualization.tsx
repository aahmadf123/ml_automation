"use client"

import React, { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { useTheme } from 'next-themes'

export interface AnimatedDataVisualizationProps {
  className?: string
  type?: 'waves' | 'particles' | 'grid' | 'nodes'
  height?: string | number
  animate?: boolean
  color?: string
  secondaryColor?: string
  density?: 'low' | 'medium' | 'high'
}

export function AnimatedDataVisualization({
  className = '',
  type = 'waves',
  height = 200,
  animate = true,
  color,
  secondaryColor,
  density = 'medium',
}: AnimatedDataVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { theme } = useTheme()
  const frameRef = useRef<number>(0)
  const isDarkMode = theme === 'dark'

  // Get colors based on theme or passed props
  const getColors = () => {
    if (color && secondaryColor) {
      return { primary: color, secondary: secondaryColor }
    }
    
    if (isDarkMode) {
      return { 
        primary: 'rgba(59, 130, 246, 0.7)', // blue-500
        secondary: 'rgba(16, 185, 129, 0.6)' // green-500
      }
    }
    
    return {
      primary: 'rgba(59, 130, 246, 0.5)', // blue-500 
      secondary: 'rgba(16, 185, 129, 0.4)' // green-500
    }
  }

  useEffect(() => {
    if (!canvasRef.current || !animate) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth
      canvas.height = typeof height === 'number' ? height : parseInt(height, 10) || 200
    }
    
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)
    
    // Drawing functions for different visualization types
    const drawFunctions = {
      waves: (timestamp: number) => {
        const { primary, secondary } = getColors()
        const width = canvas.width
        const height = canvas.height
        
        ctx.clearRect(0, 0, width, height)
        
        // First wave
        ctx.beginPath()
        const amplitude1 = height * 0.2
        const frequency1 = 0.02
        const speed1 = timestamp * 0.001
        
        ctx.moveTo(0, height * 0.5)
        for (let x = 0; x < width; x++) {
          const y = Math.sin((x * frequency1) + speed1) * amplitude1 + (height * 0.5)
          ctx.lineTo(x, y)
        }
        ctx.lineTo(width, height)
        ctx.lineTo(0, height)
        ctx.closePath()
        
        const gradient1 = ctx.createLinearGradient(0, 0, 0, height)
        gradient1.addColorStop(0, primary)
        gradient1.addColorStop(1, 'rgba(59, 130, 246, 0)')
        ctx.fillStyle = gradient1
        ctx.fill()
        
        // Second wave
        ctx.beginPath()
        const amplitude2 = height * 0.15
        const frequency2 = 0.03
        const speed2 = timestamp * 0.0015
        
        ctx.moveTo(0, height * 0.6)
        for (let x = 0; x < width; x++) {
          const y = Math.sin((x * frequency2) + speed2) * amplitude2 + (height * 0.6)
          ctx.lineTo(x, y)
        }
        ctx.lineTo(width, height)
        ctx.lineTo(0, height)
        ctx.closePath()
        
        const gradient2 = ctx.createLinearGradient(0, 0, 0, height)
        gradient2.addColorStop(0, secondary)
        gradient2.addColorStop(1, 'rgba(16, 185, 129, 0)')
        ctx.fillStyle = gradient2
        ctx.fill()
      },
      
      particles: (timestamp: number) => {
        const { primary, secondary } = getColors()
        const width = canvas.width
        const height = canvas.height
        
        // Generate particles on first run
        if (!canvasRef.current.particles) {
          const particleCount = density === 'low' ? 30 : density === 'medium' ? 50 : 80
          canvasRef.current.particles = Array.from({ length: particleCount }, () => ({
            x: Math.random() * width,
            y: Math.random() * height,
            radius: Math.random() * 3 + 1,
            color: Math.random() > 0.5 ? primary : secondary,
            speed: Math.random() * 0.5 + 0.1,
            direction: Math.random() * Math.PI * 2
          }))
        }
        
        ctx.clearRect(0, 0, width, height)
        
        // Draw and update particles
        canvasRef.current.particles.forEach((particle: any) => {
          // Update position
          particle.x += Math.cos(particle.direction) * particle.speed
          particle.y += Math.sin(particle.direction) * particle.speed
          
          // Bounce off edges
          if (particle.x < 0 || particle.x > width) {
            particle.direction = Math.PI - particle.direction
          }
          if (particle.y < 0 || particle.y > height) {
            particle.direction = -particle.direction
          }
          
          // Draw particle
          ctx.beginPath()
          ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2)
          ctx.fillStyle = particle.color
          ctx.fill()
        })
        
        // Draw connections between nearby particles
        canvasRef.current.particles.forEach((p1: any, i: number) => {
          canvasRef.current.particles.slice(i + 1).forEach((p2: any) => {
            const dx = p1.x - p2.x
            const dy = p1.y - p2.y
            const distance = Math.sqrt(dx * dx + dy * dy)
            
            if (distance < 80) {
              ctx.beginPath()
              ctx.moveTo(p1.x, p1.y)
              ctx.lineTo(p2.x, p2.y)
              ctx.strokeStyle = `rgba(125, 125, 255, ${0.2 * (1 - distance / 80)})`
              ctx.lineWidth = 0.5
              ctx.stroke()
            }
          })
        })
      },
      
      grid: (timestamp: number) => {
        const { primary, secondary } = getColors()
        const width = canvas.width
        const height = canvas.height
        
        ctx.clearRect(0, 0, width, height)
        
        // Create perspective grid effect
        const cellSize = density === 'low' ? 40 : density === 'medium' ? 30 : 20
        const perspective = timestamp * 0.0002
        
        // Draw grid lines
        for (let x = 0; x <= width; x += cellSize) {
          const offset = Math.sin(x * 0.01 + perspective) * 5
          
          ctx.beginPath()
          ctx.moveTo(x, 0)
          ctx.lineTo(x, height)
          ctx.strokeStyle = `rgba(125, 125, 255, 0.15)`
          ctx.lineWidth = 1
          ctx.stroke()
        }
        
        for (let y = 0; y <= height; y += cellSize) {
          const offset = Math.cos(y * 0.01 + perspective) * 5
          
          ctx.beginPath()
          ctx.moveTo(0, y + offset)
          ctx.lineTo(width, y)
          ctx.strokeStyle = `rgba(125, 125, 255, 0.15)`
          ctx.lineWidth = 1
          ctx.stroke()
        }
        
        // Draw animated data points
        for (let x = cellSize; x < width; x += cellSize * 3) {
          for (let y = cellSize; y < height; y += cellSize * 2) {
            const fluctuation = Math.sin(x * y * 0.0001 + perspective * 3)
            const radius = (fluctuation + 1) * 3 + 2
            
            if (Math.random() > 0.7) {
              ctx.beginPath()
              ctx.arc(x, y, radius, 0, Math.PI * 2)
              ctx.fillStyle = Math.random() > 0.5 ? primary : secondary
              ctx.fill()
            }
          }
        }
      },
      
      nodes: (timestamp: number) => {
        const { primary, secondary } = getColors()
        const width = canvas.width
        const height = canvas.height
        
        // Generate nodes on first run
        if (!canvasRef.current.nodes) {
          const nodeCount = density === 'low' ? 5 : density === 'medium' ? 8 : 12
          canvasRef.current.nodes = Array.from({ length: nodeCount }, () => ({
            x: Math.random() * width,
            y: Math.random() * height,
            radius: Math.random() * 8 + 4,
            color: Math.random() > 0.5 ? primary : secondary,
            speed: {
              x: (Math.random() - 0.5) * 0.5,
              y: (Math.random() - 0.5) * 0.5
            }
          }))
        }
        
        ctx.clearRect(0, 0, width, height)
        
        // Update and draw nodes
        canvasRef.current.nodes.forEach((node: any) => {
          // Update position
          node.x += node.speed.x
          node.y += node.speed.y
          
          // Bounce off edges
          if (node.x < node.radius || node.x > width - node.radius) {
            node.speed.x = -node.speed.x
          }
          if (node.y < node.radius || node.y > height - node.radius) {
            node.speed.y = -node.speed.y
          }
          
          // Draw node
          ctx.beginPath()
          ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
          ctx.fillStyle = node.color
          ctx.fill()
        })
        
        // Draw connections between nodes
        canvasRef.current.nodes.forEach((node1: any, i: number) => {
          canvasRef.current.nodes.forEach((node2: any, j: number) => {
            if (i !== j) {
              const dx = node1.x - node2.x
              const dy = node1.y - node2.y
              const distance = Math.sqrt(dx * dx + dy * dy)
              
              if (distance < 150) {
                ctx.beginPath()
                ctx.moveTo(node1.x, node1.y)
                ctx.lineTo(node2.x, node2.y)
                ctx.strokeStyle = `rgba(125, 125, 255, ${0.8 * (1 - distance / 150)})`
                ctx.lineWidth = 1
                ctx.stroke()
              }
            }
          })
        })
      }
    }
    
    // Animation loop
    const animate = (timestamp: number) => {
      if (drawFunctions[type]) {
        drawFunctions[type](timestamp)
      }
      
      frameRef.current = requestAnimationFrame(animate)
    }
    
    frameRef.current = requestAnimationFrame(animate)
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', resizeCanvas)
      cancelAnimationFrame(frameRef.current)
    }
  }, [animate, type, height, density, theme, color, secondaryColor])
  
  return (
    <motion.div 
      className={`relative overflow-hidden ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
    >
      <canvas 
        ref={canvasRef} 
        className="w-full"
        style={{ height: typeof height === 'number' ? `${height}px` : height }}
      />
    </motion.div>
  )
} 