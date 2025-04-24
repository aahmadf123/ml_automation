"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { TrendingUp, TrendingDown, Minus, AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"

export interface MetricData {
  title: string
  value: string | number
  change?: {
    value: number
    trend: "up" | "down" | "neutral"
    timespan: string
  }
  description?: string
  variant?: "default" | "success" | "warning" | "danger" | "info"
  icon?: React.ReactNode
  isPercentage?: boolean
  isLoading?: boolean
}

interface EnhancedMetricsCardProps {
  metric: MetricData
  className?: string
  accentColor?: string
  animated?: boolean
}

export function EnhancedMetricsCard({
  metric,
  className,
  accentColor,
  animated = true
}: EnhancedMetricsCardProps) {
  const [animatedValue, setAnimatedValue] = useState<number>(0)
  const [isVisible, setIsVisible] = useState<boolean>(false)

  const getVariantClasses = () => {
    switch (metric.variant) {
      case "success":
        return "bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800"
      case "warning":
        return "bg-yellow-50 border-yellow-200 dark:bg-yellow-950 dark:border-yellow-800"
      case "danger":
        return "bg-red-50 border-red-200 dark:bg-red-950 dark:border-red-800"
      case "info":
        return "bg-blue-50 border-blue-200 dark:bg-blue-950 dark:border-blue-800"
      default:
        return "bg-card border-border"
    }
  }

  const getTrendIcon = () => {
    if (!metric.change) return null
    
    switch (metric.change.trend) {
      case "up":
        return <TrendingUp className="h-4 w-4 text-green-500" />
      case "down":
        return <TrendingDown className="h-4 w-4 text-red-500" />
      case "neutral":
        return <Minus className="h-4 w-4 text-gray-500" />
      default:
        return null
    }
  }

  const getTrendColorClass = () => {
    if (!metric.change) return ""
    
    switch (metric.change.trend) {
      case "up":
        return "text-green-600 dark:text-green-400"
      case "down":
        return "text-red-600 dark:text-red-400"
      case "neutral":
        return "text-gray-600 dark:text-gray-400"
      default:
        return ""
    }
  }

  useEffect(() => {
    if (!animated || typeof metric.value !== 'number') return
    
    setIsVisible(true)
    
    const targetValue = typeof metric.value === 'number' ? metric.value : 0
    let startValue = 0
    
    const duration = 1500
    const frameDuration = 1000 / 60
    const totalFrames = Math.round(duration / frameDuration)
    
    let frame = 0
    const counter = setInterval(() => {
      frame++
      const progress = frame / totalFrames
      const currentValue = Math.floor(startValue + (targetValue - startValue) * progress)
      
      setAnimatedValue(currentValue)
      
      if (frame === totalFrames) {
        clearInterval(counter)
        setAnimatedValue(targetValue)
      }
    }, frameDuration)
    
    return () => clearInterval(counter)
  }, [metric.value, animated])

  if (metric.isLoading) {
    return (
      <Card className={cn("overflow-hidden", className)}>
        <CardHeader className="p-4 pb-2">
          <Skeleton className="h-5 w-1/2" />
        </CardHeader>
        <CardContent className="p-4 pt-2">
          <Skeleton className="h-10 w-3/4 mb-2" />
          <Skeleton className="h-4 w-1/3" />
        </CardContent>
      </Card>
    )
  }

  const displayValue = animated && typeof metric.value === 'number' 
    ? animatedValue 
    : metric.value
  
  const formattedValue = metric.isPercentage && typeof displayValue === 'number'
    ? `${displayValue.toFixed(1)}%`
    : displayValue

  return (
    <Card 
      className={cn(
        "overflow-hidden transition-all duration-300 border-2", 
        getVariantClasses(),
        isVisible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
        className
      )}
      style={accentColor ? { borderColor: accentColor } : {}}
    >
      {accentColor && (
        <div 
          className="h-1 w-full" 
          style={{ backgroundColor: accentColor }}
        />
      )}
      
      <CardHeader className="p-4 pb-0 flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {metric.title}
        </CardTitle>
        {metric.icon && (
          <div className="h-8 w-8 rounded-full flex items-center justify-center bg-muted/20">
            {metric.icon}
          </div>
        )}
      </CardHeader>
      
      <CardContent className="p-4">
        <div className="text-2xl font-bold mb-1">
          {formattedValue}
        </div>
        
        {metric.change && (
          <div className="flex items-center gap-1">
            {getTrendIcon()}
            <span className={cn("text-xs font-medium", getTrendColorClass())}>
              {metric.change.value > 0 ? "+" : ""}
              {metric.change.value}
              {metric.isPercentage ? "%" : ""} ({metric.change.timespan})
            </span>
          </div>
        )}
      </CardContent>
      
      {metric.description && (
        <CardFooter className="p-4 pt-0">
          <p className="text-xs text-muted-foreground">{metric.description}</p>
        </CardFooter>
      )}
    </Card>
  )
} 