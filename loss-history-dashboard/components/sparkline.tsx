"use client"

import { useEffect, useRef } from "react"

interface SparklineProps {
  data: number[]
  width?: number
  height?: number
  color?: string
  lineWidth?: number
}

export function Sparkline({ data, width = 100, height = 30, color = "#4ECDC4", lineWidth = 1.5 }: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Find min and max values for scaling
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1 // Avoid division by zero

    // Draw the sparkline
    ctx.beginPath()
    ctx.strokeStyle = color
    ctx.lineWidth = lineWidth
    ctx.lineJoin = "round"
    ctx.lineCap = "round"

    // Calculate points
    const step = width / (data.length - 1)
    const padding = 2 // Padding to avoid cutting off the line at edges

    data.forEach((value, i) => {
      const x = i * step
      // Invert the y-coordinate because canvas y increases downward
      const y = height - padding - ((value - min) / range) * (height - padding * 2)

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.stroke()

    // Add a dot at the end
    const lastX = (data.length - 1) * step
    const lastY = height - padding - ((data[data.length - 1] - min) / range) * (height - padding * 2)

    ctx.beginPath()
    ctx.fillStyle = color
    ctx.arc(lastX, lastY, 2, 0, Math.PI * 2)
    ctx.fill()
  }, [data, width, height, color, lineWidth])

  return <canvas ref={canvasRef} width={width} height={height} className="inline-block" />
}
