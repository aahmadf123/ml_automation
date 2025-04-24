"use client"

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Button } from './button'
import { cn } from '@/lib/utils'
import { ArrowRight, Play, Info, ChevronRight, Plus } from 'lucide-react'

export interface Metric {
  label: string
  value: string | number
  suffix?: string
  icon?: React.ReactNode
}

export interface HeroBannerProps {
  title: string
  subtitle?: string
  metrics?: Metric[]
  primaryAction?: {
    label: string
    onClick: () => void
    icon?: React.ReactNode
  }
  secondaryAction?: {
    label: string
    onClick: () => void
    icon?: React.ReactNode
  }
  guidedTour?: {
    label: string
    onClick: () => void
  }
  latestInsight?: {
    text: string
    link: string
  }
  className?: string
}

export function HeroBanner({
  title,
  subtitle,
  metrics = [],
  primaryAction,
  secondaryAction,
  guidedTour,
  latestInsight,
  className,
}: HeroBannerProps) {
  const [isVisible, setIsVisible] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
        }
      },
      { threshold: 0.1 }
    )
    
    if (ref.current) {
      observer.observe(ref.current)
    }
    
    return () => {
      if (ref.current) {
        observer.unobserve(ref.current)
      }
    }
  }, [])

  return (
    <div 
      ref={ref}
      className={cn(
        'relative overflow-hidden rounded-xl bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 border border-slate-200',
        className
      )}
    >
      {/* Background Pattern */}
      <div className="absolute inset-0 z-0 opacity-10">
        <svg 
          width="100%" 
          height="100%" 
          viewBox="0 0 100 100" 
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <pattern 
              id="grid" 
              width="10" 
              height="10" 
              patternUnits="userSpaceOnUse"
            >
              <path 
                d="M 10 0 L 0 0 0 10" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="0.5"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      <div className="relative z-10 px-6 py-8 md:px-10 md:py-12 lg:py-16 lg:px-16">
        <div className="max-w-5xl">
          {latestInsight && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={isVisible ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center rounded-full bg-purple-100 px-4 py-1.5 mb-6 text-sm font-medium text-purple-800"
            >
              <span className="mr-1.5 h-2 w-2 rounded-full bg-purple-500" />
              <span className="mr-2">{latestInsight.text}</span>
              <a 
                href={latestInsight.link} 
                className="inline-flex items-center font-medium text-purple-800 hover:text-purple-900"
              >
                Learn more <ChevronRight className="ml-1 h-3 w-3" />
              </a>
            </motion.div>
          )}

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={isVisible ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl md:text-5xl"
          >
            {title}
          </motion.h1>

          {subtitle && (
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={isVisible ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mt-4 max-w-2xl text-lg text-slate-600"
            >
              {subtitle}
            </motion.p>
          )}

          {(primaryAction || secondaryAction || guidedTour) && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={isVisible ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="mt-8 flex flex-wrap gap-4"
            >
              {primaryAction && (
                <Button 
                  onClick={primaryAction.onClick}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white"
                >
                  {primaryAction.label}
                  {primaryAction.icon || <ArrowRight className="ml-2 h-4 w-4" />}
                </Button>
              )}
              
              {secondaryAction && (
                <Button 
                  variant="outline" 
                  onClick={secondaryAction.onClick}
                  className="border-indigo-200 text-indigo-700 hover:bg-indigo-50"
                >
                  {secondaryAction.label}
                  {secondaryAction.icon || <Plus className="ml-2 h-4 w-4" />}
                </Button>
              )}
              
              {guidedTour && (
                <Button 
                  variant="ghost" 
                  onClick={guidedTour.onClick}
                  className="text-slate-700 hover:bg-slate-100"
                >
                  <Play className="mr-2 h-4 w-4 text-indigo-600" />
                  {guidedTour.label}
                </Button>
              )}
            </motion.div>
          )}
        </div>

        {metrics.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={isVisible ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
          >
            {metrics.map((metric, index) => (
              <AnimatedMetric 
                key={`metric-${index}`}
                metric={metric} 
                delay={0.5 + index * 0.1}
                isVisible={isVisible}
              />
            ))}
          </motion.div>
        )}
      </div>
    </div>
  )
}

function AnimatedMetric({ 
  metric, 
  delay, 
  isVisible 
}: { 
  metric: Metric, 
  delay: number,
  isVisible: boolean
}) {
  const [animatedValue, setAnimatedValue] = useState(0)
  const isNumeric = typeof metric.value === 'number'
  
  useEffect(() => {
    if (!isVisible || !isNumeric) return
    
    const targetValue = metric.value as number
    const duration = 1500
    const startTime = Date.now()
    
    const animateValue = () => {
      const now = Date.now()
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      
      // Easing function for smooth animation
      const easeOutQuart = (x: number): number => 1 - Math.pow(1 - x, 4)
      const easedProgress = easeOutQuart(progress)
      
      setAnimatedValue(Math.floor(easedProgress * targetValue))
      
      if (progress < 1) {
        requestAnimationFrame(animateValue)
      } else {
        setAnimatedValue(targetValue)
      }
    }
    
    animateValue()
  }, [isVisible, metric.value, isNumeric])

  const formatValue = (val: number | string) => {
    if (typeof val === 'string') return val
    
    // Format large numbers
    if (val >= 1000000) {
      return (val / 1000000).toFixed(1) + 'M'
    } else if (val >= 1000) {
      return (val / 1000).toFixed(1) + 'K'
    }
    
    return val.toString()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={isVisible ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay }}
      className="relative rounded-lg border border-indigo-100 bg-white/80 backdrop-blur-sm p-6 shadow-sm"
    >
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-500">{metric.label}</p>
        {metric.icon && (
          <div className="h-8 w-8 rounded-full bg-indigo-100 p-1.5 text-indigo-600">
            {metric.icon}
          </div>
        )}
      </div>
      <div className="mt-2 flex items-baseline">
        <p className="text-3xl font-bold text-slate-900">
          {isNumeric ? formatValue(animatedValue) : formatValue(metric.value)}
        </p>
        {metric.suffix && (
          <p className="ml-1 text-sm font-medium text-slate-500">{metric.suffix}</p>
        )}
      </div>
      
      {/* Decorative element */}
      <div className="absolute bottom-0 right-0 h-16 w-16 translate-x-1/3 translate-y-1/3 rounded-full bg-indigo-200 opacity-20" />
    </motion.div>
  )
} 