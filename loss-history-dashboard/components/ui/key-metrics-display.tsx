"use client"

import React from 'react'
import { motion } from 'framer-motion'
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  TimerIcon, 
  AlertCircle, 
  Code2, 
  CheckCircle2, 
  AlertTriangle 
} from 'lucide-react'

import { Card, CardContent } from './card'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './tooltip'
import { cn } from '@/lib/utils'
import { Badge } from './badge'
import { Skeleton } from './skeleton'

export interface KeyMetric {
  id: string
  title: string
  value: string | number
  valuePrefix?: string
  valueSuffix?: string
  change?: number
  status?: 'positive' | 'negative' | 'neutral' | 'warning'
  icon?: React.ReactNode
  tooltip?: string
  isPercentage?: boolean
  animate?: boolean
}

export interface KeyMetricsDisplayProps {
  metrics: KeyMetric[]
  isLoading?: boolean
  className?: string
  layout?: 'grid' | 'flex'
  showHeader?: boolean
  title?: string
  subtitle?: string
}

export function KeyMetricsDisplay({
  metrics,
  isLoading = false,
  className = '',
  layout = 'grid',
  showHeader = false,
  title = 'Key Project Metrics',
  subtitle = 'Real-time metrics from your ML project',
}: KeyMetricsDisplayProps) {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }
  
  const item = {
    hidden: { y: 20, opacity: 0 },
    show: { y: 0, opacity: 1 }
  }
  
  const getStatusColor = (status: KeyMetric['status']) => {
    switch (status) {
      case 'positive':
        return 'text-green-500 dark:text-green-400'
      case 'negative':
        return 'text-red-500 dark:text-red-400'
      case 'warning':
        return 'text-amber-500 dark:text-amber-400'
      default:
        return 'text-blue-500 dark:text-blue-400'
    }
  }
  
  const getDefaultIcon = (status: KeyMetric['status']) => {
    switch (status) {
      case 'positive':
        return <CheckCircle2 className="h-5 w-5" />
      case 'negative':
        return <AlertCircle className="h-5 w-5" />
      case 'warning':
        return <AlertTriangle className="h-5 w-5" />
      default:
        return <BarChart3 className="h-5 w-5" />
    }
  }
  
  const getChangeIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="h-4 w-4 text-green-500" />
    if (change < 0) return <TrendingDown className="h-4 w-4 text-red-500" />
    return null
  }
  
  const formatChange = (change: number, isPercentage: boolean = false) => {
    const prefix = change > 0 ? '+' : ''
    const value = `${prefix}${change.toFixed(isPercentage ? 1 : 0)}${isPercentage ? '%' : ''}`
    return value
  }
  
  return (
    <div className={className}>
      {showHeader && (
        <div className="mb-6">
          <motion.h2 
            className="text-2xl font-bold tracking-tight"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            {title}
          </motion.h2>
          <motion.p 
            className="text-sm text-muted-foreground"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            {subtitle}
          </motion.p>
        </div>
      )}
      
      <motion.div 
        className={cn(
          layout === 'grid' 
            ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4' 
            : 'flex flex-wrap gap-4'
        )}
        variants={container}
        initial="hidden"
        animate="show"
      >
        {isLoading ? (
          // Loading skeletons
          Array.from({ length: 4 }).map((_, i) => (
            <Card key={`skeleton-${i}`} className="overflow-hidden">
              <CardContent className="p-4">
                <Skeleton className="h-4 w-24 mb-2" />
                <Skeleton className="h-8 w-16 mb-2" />
                <Skeleton className="h-4 w-32" />
              </CardContent>
            </Card>
          ))
        ) : (
          // Actual metrics
          metrics.map((metric) => (
            <motion.div key={metric.id} variants={item}>
              <TooltipProvider>
                <Tooltip delayDuration={300}>
                  <TooltipTrigger asChild>
                    <Card className={cn(
                      "relative overflow-hidden transition-shadow hover:shadow-md cursor-pointer",
                      metric.status && `border-l-4 border-l-${getStatusColor(metric.status).replace('text-', '')}`
                    )}>
                      <CardContent className="p-4">
                        <div className="flex justify-between items-start">
                          <h3 className="text-sm font-medium text-muted-foreground">
                            {metric.title}
                          </h3>
                          <span className={cn(
                            "p-1 rounded-full", 
                            getStatusColor(metric.status)
                          )}>
                            {metric.icon || getDefaultIcon(metric.status)}
                          </span>
                        </div>
                        
                        <div className="mt-2">
                          <motion.div 
                            className="text-2xl font-bold"
                            initial={metric.animate ? { opacity: 0, y: 20 } : undefined}
                            animate={metric.animate ? { opacity: 1, y: 0 } : undefined}
                            transition={{ type: 'spring', stiffness: 100 }}
                          >
                            {metric.valuePrefix && (
                              <span className="text-xl text-muted-foreground mr-1">
                                {metric.valuePrefix}
                              </span>
                            )}
                            {metric.value}
                            {metric.valueSuffix && (
                              <span className="text-xl text-muted-foreground ml-1">
                                {metric.valueSuffix}
                              </span>
                            )}
                          </motion.div>
                          
                          {metric.change !== undefined && (
                            <div className="flex items-center mt-1 text-sm">
                              {getChangeIcon(metric.change)}
                              <span className={cn(
                                "ml-1",
                                metric.change > 0 ? "text-green-500" : 
                                metric.change < 0 ? "text-red-500" : "text-gray-500"
                              )}>
                                {formatChange(metric.change, metric.isPercentage)}
                              </span>
                              <span className="text-muted-foreground ml-1 text-xs">
                                vs last period
                              </span>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </TooltipTrigger>
                  {metric.tooltip && (
                    <TooltipContent side="bottom">
                      <p className="max-w-xs">{metric.tooltip}</p>
                    </TooltipContent>
                  )}
                </Tooltip>
              </TooltipProvider>
            </motion.div>
          ))
        )}
      </motion.div>
    </div>
  )
} 