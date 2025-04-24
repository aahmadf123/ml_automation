"use client"

import React, { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronLeft, ChevronRight, BellRing, Lightbulb } from "lucide-react"
import { Button } from "./button"

interface Insight {
  id: string
  title: string
  description: string
  type: "update" | "alert" | "tip" | "news"
  date: string
  url?: string
}

interface LatestInsightsBannerProps {
  insights?: Insight[]
  autoRotate?: boolean
  rotationInterval?: number
  className?: string
}

// Default insights data
const defaultInsights: Insight[] = [
  {
    id: "1",
    title: "Model drift detected in production",
    description: "Anomalies detected in model #23C. We recommend reviewing recent predictions.",
    type: "alert",
    date: "2 hours ago",
    url: "/incidents/latest"
  },
  {
    id: "2",
    title: "New training dataset available",
    description: "Updated dataset with 12k new labeled examples is ready for your next training run.",
    type: "update",
    date: "Yesterday",
    url: "/data-ingestion"
  },
  {
    id: "3",
    title: "Feature importance visualization updated",
    description: "Check out the improved model explainability dashboard with SHAP integration.",
    type: "tip",
    date: "2 days ago",
    url: "/model-explainability"
  },
  {
    id: "4",
    title: "Latest model showed 8.4% improvement",
    description: "The newly deployed recommendation model is outperforming the previous version.",
    type: "news",
    date: "3 days ago",
    url: "/model-metrics"
  }
]

export function LatestInsightsBanner({
  insights = defaultInsights,
  autoRotate = true,
  rotationInterval = 5000,
  className = ""
}: LatestInsightsBannerProps) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isPaused, setIsPaused] = useState(false)

  const nextInsight = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % insights.length)
  }

  const prevInsight = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + insights.length) % insights.length)
  }

  useEffect(() => {
    if (autoRotate && !isPaused) {
      const interval = setInterval(nextInsight, rotationInterval)
      return () => clearInterval(interval)
    }
  }, [autoRotate, rotationInterval, isPaused, insights.length])

  const getIconByType = (type: string) => {
    switch (type) {
      case "alert":
        return <BellRing className="h-5 w-5 text-orange-500" />
      case "tip":
        return <Lightbulb className="h-5 w-5 text-yellow-500" />
      case "update":
        return (
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" 
            className="h-5 w-5 text-blue-500" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
        )
      case "news":
        return (
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" 
            className="h-5 w-5 text-green-500" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
          </svg>
        )
      default:
        return null
    }
  }

  // Background colors by type
  const getBgColorByType = (type: string) => {
    switch (type) {
      case "alert":
        return "bg-orange-50 dark:bg-orange-950/30"
      case "tip":
        return "bg-yellow-50 dark:bg-yellow-950/30"
      case "update":
        return "bg-blue-50 dark:bg-blue-950/30"
      case "news":
        return "bg-green-50 dark:bg-green-950/30"
      default:
        return "bg-gray-50 dark:bg-gray-800/30"
    }
  }

  const currentInsight = insights[currentIndex]

  return (
    <div 
      className={`relative overflow-hidden rounded-lg border ${getBgColorByType(currentInsight.type)} ${className}`}
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      <div className="flex items-center justify-between">
        <Button 
          variant="ghost" 
          size="icon" 
          className="z-10 h-8 w-8 rounded-full opacity-70 hover:opacity-100" 
          onClick={prevInsight}
        >
          <ChevronLeft className="h-4 w-4" />
          <span className="sr-only">Previous</span>
        </Button>
        
        <div className="flex-1 overflow-hidden px-1">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentInsight.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="py-3"
            >
              <div className="flex items-center gap-4 px-2">
                <div className="flex-shrink-0 rounded-full p-1.5">
                  {getIconByType(currentInsight.type)}
                </div>
                
                <div className="flex-1 min-w-0">
                  <p className="truncate text-sm font-medium">
                    {currentInsight.title}
                  </p>
                  <p className="truncate text-xs text-muted-foreground">
                    {currentInsight.description}
                  </p>
                </div>
                
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground whitespace-nowrap">
                    {currentInsight.date}
                  </span>
                  
                  {currentInsight.url && (
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="h-7 px-2 text-xs"
                      asChild
                    >
                      <a href={currentInsight.url}>View</a>
                    </Button>
                  )}
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
        
        <Button 
          variant="ghost" 
          size="icon" 
          className="z-10 h-8 w-8 rounded-full opacity-70 hover:opacity-100" 
          onClick={nextInsight}
        >
          <ChevronRight className="h-4 w-4" />
          <span className="sr-only">Next</span>
        </Button>
      </div>
      
      {/* Progress bar */}
      {autoRotate && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-background/20">
          <motion.div 
            className="h-full bg-primary/50"
            initial={{ width: "0%" }}
            animate={{ width: "100%" }}
            transition={{ 
              duration: rotationInterval / 1000,
              ease: "linear",
              repeat: Infinity,
              repeatType: "loop"
            }}
            key={currentIndex}
          />
        </div>
      )}
      
      {/* Pagination dots */}
      <div className="absolute bottom-1 left-1/2 -translate-x-1/2 flex gap-1">
        {insights.map((_, index) => (
          <button
            key={index}
            className={`h-1.5 rounded-full ${index === currentIndex ? 'w-3 bg-primary' : 'w-1.5 bg-primary/30'}`}
            onClick={() => setCurrentIndex(index)}
          />
        ))}
      </div>
    </div>
  )
} 