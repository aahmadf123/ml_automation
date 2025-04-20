"use client"

import { useEffect, useRef } from "react"

interface MiniSparklineProps {
  data: number[]
  color: string
  className?: string
  inverted?: boolean
}

export function MiniSparkline({ data, color, className = "", inverted = false }: MiniSparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set up dimensions
    const width = canvas.width
    const height = canvas.height
    const padding = 2
    const effectiveWidth = width - padding * 2
    const effectiveHeight = height - padding * 2

    // Find min and max values
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min

    // Draw the sparkline
    ctx.beginPath()
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5

    data.forEach((value, index) => {
      const x = padding + (index / (data.length - 1)) * effectiveWidth

      // If inverted, higher values go down instead of up
      let normalizedValue
      if (inverted) {
        normalizedValue = range === 0 ? 0.5 : (max - value) / range
      } else {
        normalizedValue = range === 0 ? 0.5 : (value - min) / range
      }

      const y = padding + normalizedValue * effectiveHeight

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.stroke()
  }, [data, color, inverted])

  return <canvas ref={canvasRef} width={60} height={20} className={className} />
}
