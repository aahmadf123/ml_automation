"use client"

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { cn } from "@/lib/utils"
import { ArrowDownIcon, ArrowUpIcon } from 'lucide-react'

interface MetricProps {
  label: string
  value: number
  suffix?: string
  prefix?: string
  trend?: number
  icon?: React.ReactNode
  className?: string
}

export function AnimatedMetrics({
  metrics,
  className
}: {
  metrics: MetricProps[]
  className?: string
}) {
  return (
    <div className={cn("grid gap-4 sm:grid-cols-2 lg:grid-cols-3", className)}>
      {metrics.map((metric, index) => (
        <Metric 
          key={index}
          label={metric.label}
          value={metric.value}
          suffix={metric.suffix}
          prefix={metric.prefix}
          trend={metric.trend}
          icon={metric.icon}
        />
      ))}
    </div>
  )
}

function Metric({
  label,
  value,
  suffix = "",
  prefix = "",
  trend,
  icon,
  className
}: MetricProps) {
  const [displayValue, setDisplayValue] = useState(0)
  
  useEffect(() => {
    const duration = 1500 // milliseconds
    const steps = 30
    const stepDuration = duration / steps
    const increment = value / steps
    let current = 0
    let timer: NodeJS.Timeout

    const updateValue = () => {
      current += increment
      if (current >= value) {
        current = value
        setDisplayValue(current)
        clearInterval(timer)
      } else {
        setDisplayValue(current)
      }
    }

    timer = setInterval(updateValue, stepDuration)
    
    return () => {
      clearInterval(timer)
    }
  }, [value])

  const isTrendPositive = (trend || 0) > 0
  const isTrendNegative = (trend || 0) < 0
  
  const formatValue = (val: number) => {
    if (Number.isInteger(val)) {
      return val.toLocaleString()
    } else {
      return val.toFixed(1)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 * (suffix ? 1 : 0) }}
      className={cn(
        "p-4 rounded-lg border border-slate-200 bg-white/50 backdrop-blur-sm hover:shadow-md transition-shadow",
        className
      )}
    >
      <div className="flex justify-between items-start">
        <div className="space-y-0.5">
          <p className="text-sm font-medium text-slate-500">{label}</p>
          <h4 className="text-2xl font-bold text-slate-800">
            {prefix}{formatValue(displayValue)}{suffix}
          </h4>
        </div>
        {icon && (
          <div className="rounded-full bg-blue-50 p-2 text-blue-600">
            {icon}
          </div>
        )}
      </div>
      
      {trend !== undefined && (
        <div className="mt-2 flex items-center text-sm">
          <span 
            className={cn(
              "flex items-center font-medium",
              isTrendPositive ? "text-green-600" : "",
              isTrendNegative ? "text-red-600" : "",
              !isTrendPositive && !isTrendNegative ? "text-slate-600" : ""
            )}
          >
            {isTrendPositive && <ArrowUpIcon className="mr-1 h-3 w-3" />}
            {isTrendNegative && <ArrowDownIcon className="mr-1 h-3 w-3" />}
            {trend > 0 && "+"}
            {trend}%
          </span>
          <span className="ml-1 text-slate-500">vs. last month</span>
        </div>
      )}
    </motion.div>
  )
} 