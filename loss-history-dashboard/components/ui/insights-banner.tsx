"use client"

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from './button'
import { ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface Insight {
  id: string
  title: string
  description: string
  link?: string
  type?: 'info' | 'success' | 'warning' | 'new'
}

export interface InsightsBannerProps {
  insights: Insight[]
  autoRotate?: boolean
  interval?: number
  className?: string
}

export function InsightsBanner({
  insights,
  autoRotate = true,
  interval = 6000,
  className,
}: InsightsBannerProps) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [paused, setPaused] = useState(false)

  const goToPrevious = () => {
    setCurrentIndex((prev) => (prev === 0 ? insights.length - 1 : prev - 1))
  }

  const goToNext = () => {
    setCurrentIndex((prev) => (prev === insights.length - 1 ? 0 : prev + 1))
  }

  // Auto-rotate insights
  useEffect(() => {
    if (!autoRotate || paused || insights.length <= 1) return
    
    const timer = setInterval(goToNext, interval)
    
    return () => {
      clearInterval(timer)
    }
  }, [autoRotate, currentIndex, interval, paused, insights.length])

  // Return null if no insights
  if (!insights.length) return null
  
  const getTypeStyles = (type: Insight['type'] = 'info') => {
    switch (type) {
      case 'success':
        return 'bg-green-50 border-green-200 text-green-800'
      case 'warning':
        return 'bg-amber-50 border-amber-200 text-amber-800'
      case 'new':
        return 'bg-purple-50 border-purple-200 text-purple-800'
      case 'info':
      default:
        return 'bg-blue-50 border-blue-200 text-blue-800'
    }
  }
  
  const getTypeIndicator = (type: Insight['type'] = 'info') => {
    switch (type) {
      case 'success':
        return 'bg-green-500'
      case 'warning':
        return 'bg-amber-500'
      case 'new':
        return 'bg-purple-500'
      case 'info':
      default:
        return 'bg-blue-500'
    }
  }

  return (
    <div 
      className={cn(
        'relative rounded-lg border shadow-sm overflow-hidden',
        getTypeStyles(insights[currentIndex].type),
        className
      )}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >
      <div className="px-4 py-3 sm:px-6">
        <div className="flex flex-wrap items-center justify-between">
          <div className="flex items-center space-x-3 min-w-0 flex-1">
            <span className={cn('h-2 w-2 flex-shrink-0 rounded-full', getTypeIndicator(insights[currentIndex].type))} />
            
            <AnimatePresence mode="wait">
              <motion.div
                key={insights[currentIndex].id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="min-w-0 flex-1"
              >
                <div className="flex flex-col sm:flex-row sm:items-center">
                  <p className="truncate font-medium mr-2">
                    {insights[currentIndex].title}
                  </p>
                  <p className="truncate text-sm opacity-80">
                    {insights[currentIndex].description}
                  </p>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
          
          <div className="flex-shrink-0 flex items-center mt-2 sm:mt-0 sm:ml-4">
            {insights[currentIndex].link && (
              <Button
                variant="ghost"
                size="sm"
                asChild
                className="mr-2 hover:bg-white/30"
              >
                <a
                  href={insights[currentIndex].link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center"
                >
                  View details
                  <ExternalLink className="ml-1 h-3 w-3" />
                </a>
              </Button>
            )}
            
            {insights.length > 1 && (
              <div className="flex items-center">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={goToPrevious}
                  className="h-8 w-8 hover:bg-white/30"
                >
                  <ChevronLeft className="h-4 w-4" />
                  <span className="sr-only">Previous</span>
                </Button>
                
                <span className="mx-1 text-xs opacity-70">
                  {currentIndex + 1}/{insights.length}
                </span>
                
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={goToNext}
                  className="h-8 w-8 hover:bg-white/30"
                >
                  <ChevronRight className="h-4 w-4" />
                  <span className="sr-only">Next</span>
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Progress bar */}
      {autoRotate && insights.length > 1 && (
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/5">
          <motion.div
            className={cn("h-full", getTypeIndicator(insights[currentIndex].type))}
            initial={{ width: "0%" }}
            animate={{ width: paused ? "var(--progress-width)" : "100%" }}
            transition={{
              duration: paused ? 0 : interval / 1000,
              ease: "linear",
            }}
            onAnimationComplete={() => {
              if (!paused) goToNext()
            }}
            style={{ 
              "--progress-width": paused ? `${(Date.now() % interval) / interval * 100}%` : undefined 
            } as React.CSSProperties}
          />
        </div>
      )}
    </div>
  )
} 