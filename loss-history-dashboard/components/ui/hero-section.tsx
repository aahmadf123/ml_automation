"use client"

import React, { useState, useEffect, useRef } from "react"
import { motion, useAnimation } from "framer-motion"
import { useInView } from "react-intersection-observer"
import { Button } from "./button"
import { 
  BarChart3, 
  LineChart, 
  Users, 
  Activity, 
  AlertTriangle, 
  RefreshCw,
  ArrowRight,
  TrendingUp,
  ArrowDownRight
} from "lucide-react"

interface MetricItem {
  label: string
  value: string
  icon: React.ReactNode
  change?: {
    value: string
    positive: boolean
  }
}

interface HeroSectionProps {
  title?: string
  description?: string
  metrics?: MetricItem[]
  primaryActionLabel?: string
  primaryActionHref?: string
  secondaryActionLabel?: string
  secondaryActionHref?: string
}

export function HeroSection({
  title = "Model Comparison Dashboard",
  description = "Compare the traditional model with 48 attributes to our enhanced fast decay model with significantly improved predictive power.",
  metrics,
  primaryActionLabel = "View Detailed Comparison",
  primaryActionHref = "/model-comparison",
  secondaryActionLabel = "See Performance Metrics",
  secondaryActionHref = "/model-metrics",
}: HeroSectionProps) {
  const controls = useAnimation()
  const { ref, inView } = useInView({ 
    threshold: 0.2,
    triggerOnce: true 
  })

  useEffect(() => {
    if (inView) {
      controls.start("visible")
    }
  }, [controls, inView])

  const defaultMetrics: MetricItem[] = [
    {
      label: "Traditional Model",
      value: "67%",
      icon: <BarChart3 className="h-4 w-4" />,
      change: {
        value: "R² Score",
        positive: false
      }
    },
    {
      label: "Enhanced Model",
      value: "79%",
      icon: <Activity className="h-4 w-4" />,
      change: {
        value: "R² Score",
        positive: true
      }
    },
    {
      label: "R² Improvement",
      value: "+17.9%",
      icon: <TrendingUp className="h-4 w-4" />,
      change: {
        value: "Better",
        positive: true
      }
    },
    {
      label: "Error Reduction",
      value: "26%",
      icon: <ArrowDownRight className="h-4 w-4" />,
      change: {
        value: "Lower RMSE",
        positive: true
      }
    }
  ]

  const displayMetrics = metrics || defaultMetrics

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.2
      }
    }
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 }
    }
  }

  return (
    <section className="relative overflow-hidden bg-background py-24" ref={ref}>
      {/* Background Elements */}
      <div className="absolute inset-0 z-0 opacity-5 pointer-events-none overflow-hidden">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      <motion.div
        className="absolute right-[-10%] top-[15%] w-[500px] h-[500px] rounded-full bg-primary/10 blur-3xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.6 }}
        transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
      />

      <div className="container mx-auto px-4 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Content Column */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate={controls}
            className="max-w-2xl"
          >
            <motion.h1 
              variants={itemVariants}
              className="text-4xl md:text-5xl font-bold tracking-tight mb-6"
            >
              {title}
            </motion.h1>
            
            <motion.p 
              variants={itemVariants}
              className="text-xl text-muted-foreground mb-8"
            >
              {description}
            </motion.p>
            
            <motion.div 
              variants={itemVariants}
              className="flex flex-wrap gap-4 mb-12"
            >
              <Button 
                size="lg" 
                asChild
                className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70"
              >
                <a href={primaryActionHref}>
                  {primaryActionLabel}
                  <ArrowRight className="ml-2 h-4 w-4" />
                </a>
              </Button>
              <Button variant="outline" size="lg" asChild>
                <a href={secondaryActionHref}>
                  {secondaryActionLabel}
                </a>
              </Button>
            </motion.div>
            
            {/* Metrics */}
            <motion.div 
              variants={containerVariants} 
              className="grid grid-cols-2 gap-6"
            >
              {displayMetrics.map((metric, idx) => (
                <motion.div 
                  key={idx} 
                  variants={itemVariants}
                  className="bg-card p-4 rounded-lg border shadow-sm"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-muted-foreground">
                      {metric.label}
                    </span>
                    <span className="p-1.5 rounded-full bg-primary/10">
                      {metric.icon}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-bold">{metric.value}</span>
                    {metric.change && (
                      <span className={`text-xs font-medium ${metric.change.positive ? 'text-green-500' : 'text-red-500'}`}>
                        {metric.change.value}
                      </span>
                    )}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>
          
          {/* Visualization Column */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="hidden lg:block"
          >
            <div className="relative">
              <AnimatedVisualization />
              
              {/* Floating Metrics Overlay */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1.5 }}
                className="absolute -top-12 right-0 bg-card p-4 rounded-lg border shadow-md"
              >
                <div className="flex gap-3 items-center">
                  <div className="p-2 bg-primary/10 rounded-full">
                    <LineChart className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <div className="text-sm font-medium">Model Performance</div>
                    <div className="text-2xl font-bold">+12.4%</div>
                  </div>
                </div>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1.8 }}
                className="absolute bottom-12 -left-8 bg-card p-4 rounded-lg border shadow-md"
              >
                <div className="flex gap-3 items-center">
                  <div className="p-2 bg-green-100 rounded-full dark:bg-green-900/30">
                    <Users className="h-5 w-5 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <div className="text-sm font-medium">User Satisfaction</div>
                    <div className="text-2xl font-bold">96%</div>
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

const AnimatedVisualization = () => {
  const [data, setData] = useState<number[]>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  // Generate random data for the visualization
  useEffect(() => {
    const initialData = Array.from({ length: 12 }, () => 
      Math.floor(Math.random() * 50) + 30
    )
    setData(initialData)
    
    // Update data periodically
    const interval = setInterval(() => {
      setData(prevData => {
        const newData = [...prevData]
        // Modify a random bar
        const idx = Math.floor(Math.random() * newData.length)
        newData[idx] = Math.floor(Math.random() * 50) + 30
        return newData
      })
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])
  
  // Draw the visualization
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Set canvas dimensions
    canvas.width = 500
    canvas.height = 300
    
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw the chart
    const barWidth = canvas.width / data.length / 2
    const spacing = barWidth / 2
    
    // Draw grid
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 0.5
    
    for (let i = 0; i < 5; i++) {
      const y = canvas.height - (i * canvas.height / 4)
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }
    
    // Draw the bars
    data.forEach((value, index) => {
      const x = index * (barWidth + spacing) * 2 + spacing
      const height = (value / 100) * canvas.height
      const y = canvas.height - height
      
      // Create gradient
      const gradient = ctx.createLinearGradient(x, y, x, canvas.height)
      gradient.addColorStop(0, 'rgba(124, 58, 237, 0.8)')  // Primary color (purple)
      gradient.addColorStop(1, 'rgba(124, 58, 237, 0.2)')
      
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.roundRect(x, y, barWidth, height, [4, 4, 0, 0])
      ctx.fill()
      
      // Add a line on top
      ctx.strokeStyle = 'rgba(124, 58, 237, 1)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(x, y)
      ctx.lineTo(x + barWidth, y)
      ctx.stroke()
    })
    
    // Connect the bars with a line
    ctx.strokeStyle = 'rgba(124, 58, 237, 0.6)'
    ctx.lineWidth = 2
    ctx.beginPath()
    
    data.forEach((value, index) => {
      const x = index * (barWidth + spacing) * 2 + spacing + barWidth / 2
      const height = (value / 100) * canvas.height
      const y = canvas.height - height
      
      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    
    ctx.stroke()
    
    // Add dots at data points
    data.forEach((value, index) => {
      const x = index * (barWidth + spacing) * 2 + spacing + barWidth / 2
      const height = (value / 100) * canvas.height
      const y = canvas.height - height
      
      ctx.fillStyle = '#ffffff'
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fill()
      
      ctx.strokeStyle = 'rgba(124, 58, 237, 1)'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.stroke()
    })
    
  }, [data])

  return (
    <div className="relative bg-card rounded-xl shadow-lg p-6 border overflow-hidden">
      <canvas 
        ref={canvasRef} 
        className="w-full h-auto"
      />
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent to-background/5"></div>
    </div>
  )
} 