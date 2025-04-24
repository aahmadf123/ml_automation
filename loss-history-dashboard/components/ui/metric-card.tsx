"use client"

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  ArrowDownIcon, 
  ArrowUpIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  BarChart3Icon,
  Percent,
  Clock,
  AlertCircle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from "./tooltip"

const metricCardVariants = cva(
  'relative overflow-hidden rounded-lg border p-4 shadow-sm transition-all hover:shadow-md',
  {
    variants: {
      variant: {
        default: 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700',
        primary: 'bg-blue-50 dark:bg-blue-900/20 border-blue-100 dark:border-blue-800',
        success: 'bg-green-50 dark:bg-green-900/20 border-green-100 dark:border-green-800',
        warning: 'bg-amber-50 dark:bg-amber-900/20 border-amber-100 dark:border-amber-800',
        danger: 'bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800',
        info: 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-100 dark:border-indigo-800',
      },
      isLoading: {
        true: 'opacity-70 pointer-events-none',
        false: ''
      }
    },
    defaultVariants: {
      variant: 'default',
      isLoading: false
    }
  }
)

export interface MetricCardProps extends VariantProps<typeof metricCardVariants> {
  title: string
  value: string | number
  icon?: React.ReactNode
  trend?: number
  trendLabel?: string
  previousValue?: string | number
  previousPeriod?: string
  description?: string
  isLoading?: boolean
  animate?: boolean
  className?: string
  onClick?: () => void
  change?: number
  tooltip?: string
}

export function MetricCard({
  title,
  value,
  icon,
  trend,
  trendLabel,
  previousValue,
  previousPeriod,
  description,
  isLoading = false,
  animate = true,
  variant = 'default',
  className,
  onClick,
  change,
  tooltip,
}: MetricCardProps) {
  const [hasAnimated, setHasAnimated] = useState(false)
  const [displayValue, setDisplayValue] = useState(0)
  const numericValue = typeof value === 'string' ? parseFloat(value) || 0 : value
  
  // Animation for numeric values
  useEffect(() => {
    if (!animate || isLoading || isNaN(numericValue)) {
      setDisplayValue(numericValue)
      return
    }

    let start = 0
    const end = numericValue
    const duration = 1000
    const startTime = Date.now()

    const animate = () => {
      const now = Date.now()
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      
      // Use easeOutQuart for smooth animation that slows down at the end
      const easeProgress = 1 - Math.pow(1 - progress, 4)
      const current = Math.floor(start + (end - start) * easeProgress)
      
      setDisplayValue(current)
      
      if (progress < 1) {
        requestAnimationFrame(animate)
      } else {
        setDisplayValue(end)
      }
    }
    
    requestAnimationFrame(animate)
  }, [numericValue, animate, isLoading])

  // Animate value on mount
  useEffect(() => {
    if (animate && typeof value === "number" && !hasAnimated) {
      const duration = 1500
      const startTime = Date.now()
      
      const animateValue = () => {
        const now = Date.now()
        const elapsed = now - startTime
        const progress = Math.min(elapsed / duration, 1)
        
        // Easing function for smoother animation
        const easeOutQuart = (x: number) => 1 - Math.pow(1 - x, 4)
        const easedProgress = easeOutQuart(progress)
        
        setDisplayValue(Math.floor(easedProgress * numericValue))
        
        if (progress < 1) {
          requestAnimationFrame(animateValue)
        } else {
          setHasAnimated(true)
        }
      }
      
      requestAnimationFrame(animateValue)
    }
  }, [value, animate, hasAnimated, numericValue])

  const formattedValue = typeof value === 'string' 
    ? value 
    : new Intl.NumberFormat('en-US').format(Math.round(displayValue))

  // Determine if trend is positive or negative
  const isTrendPositive = trend && trend > 0
  const isTrendNegative = trend && trend < 0
  
  // Choose trend icon
  const getTrendIcon = () => {
    if (!trend) return null
    
    if (isTrendPositive) {
      return variant === 'danger' 
        ? <TrendingUpIcon className="h-3 w-3" />
        : <TrendingUpIcon className="h-3 w-3" />
    }
    
    return variant === 'success' 
      ? <TrendingDownIcon className="h-3 w-3" />
      : <TrendingDownIcon className="h-3 w-3" />
  }
  
  // Dynamic classes for trend
  const getTrendClasses = () => {
    if (!trend) return ''
    
    if (isTrendPositive) {
      return variant === 'danger' 
        ? 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
        : 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30'
    }
    
    return variant === 'success'
      ? 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30'
      : 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
  }

  // Determine card styling based on variant
  const getVariantStyles = () => {
    switch (variant) {
      case "success":
        return "bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-800/30"
      case "danger":
        return "bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-800/30"
      case "warning":
        return "bg-amber-50 dark:bg-amber-950/20 border-amber-200 dark:border-amber-800/30"
      case "info":
        return "bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800/30"
      default:
        return "bg-card border-muted"
    }
  }

  // Get icon color class based on variant
  const getIconColorClass = () => {
    switch (variant) {
      case "success":
        return "text-green-500 dark:text-green-400"
      case "danger":
        return "text-red-500 dark:text-red-400"
      case "warning":
        return "text-amber-500 dark:text-amber-400"
      case "info":
        return "text-blue-500 dark:text-blue-400"
      default:
        return "text-primary"
    }
  }

  // Render change indicator and value
  const renderChangeIndicator = () => {
    if (change === undefined || change === 0) {
      return (
        <div className="flex items-center text-muted-foreground">
          <Minus className="mr-1 h-3 w-3" />
          <span className="text-xs">No change</span>
        </div>
      )
    }

    const isPositive = change > 0
    const absChange = Math.abs(change)
    const Icon = isPositive ? TrendingUp : TrendingDown
    const colorClass = isPositive 
      ? "text-green-600 dark:text-green-400" 
      : "text-red-600 dark:text-red-400"

    return (
      <div className={`flex items-center ${colorClass}`}>
        <Icon className="mr-1 h-3 w-3" />
        <span className="text-xs font-medium">{isPositive ? "+" : ""}{change}%</span>
      </div>
    )
  }

  const cardContent = (
    <>
      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute inset-0 bg-background/50 backdrop-blur-[1px] z-10 flex items-center justify-center">
          <motion.div
            className="h-4 w-4 rounded-full border-2 border-primary/30 border-t-primary"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          />
        </div>
      )}
      
      {/* Background pattern */}
      <div className="absolute right-0 top-0 opacity-5 pointer-events-none">
        <svg width="140" height="140" viewBox="0 0 140 140" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path 
            d="M0 0H140V140H0V0Z" 
            fill="url(#metric-card-pattern)" 
            fillOpacity="0.2"
          />
          <defs>
            <pattern id="metric-card-pattern" patternContentUnits="objectBoundingBox" width="0.2" height="0.2">
              <use href="#metric-card-wave" transform="scale(0.00714)" />
            </pattern>
            <image id="metric-card-wave" width="28" height="28" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCAyOCAyOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTE0IDI4QzIxLjczMiAyOCAyOCAyMS43MzIgMjggMTRDMjggNi4yNjgwMSAyMS43MzIgMCAxNCAwQzYuMjY4MDEgMCAwIDYuMjY4MDEgMCAxNEMwIDIxLjczMiA2LjI2ODAxIDI4IDE0IDI4Wk0xNCAxOC4yQzE2LjMxOTYgMTguMiAxOC4yIDE2LjMxOTYgMTguMiAxNEMxOC4yIDExLjY4MDQgMTYuMzE5NiA5LjggMTQgOS44QzExLjY4MDQgOS44IDkuOCAxMS42ODA0IDkuOCAxNEM5LjggMTYuMzE5NiAxMS42ODA0IDE4LjIgMTQgMTguMloiIGZpbGw9ImN1cnJlbnRDb2xvciIvPjwvc3ZnPg==" />
          </defs>
        </svg>
      </div>
      
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          {icon && (
            <span className={`rounded-full p-1.5 ${getIconColorClass()}`}>
              {icon}
            </span>
          )}
        </div>
        
        <div className="space-y-1">
          <div className="flex items-baseline gap-2">
            <motion.p 
              className="text-2xl font-semibold"
              key={`${value}`}
              initial={animate ? { opacity: 0, scale: 0.5 } : {}}
              animate={animate ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              {formattedValue}
            </motion.p>
            
            {trend !== undefined && (
              <div 
                className={cn(
                  "inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-xs font-medium",
                  getTrendClasses()
                )}
              >
                {getTrendIcon()}
                <span>{Math.abs(trend)}%</span>
              </div>
            )}
          </div>
        
          {trendLabel && (
            <p className="text-xs text-muted-foreground">
              {trendLabel}
            </p>
          )}
          
          {previousValue !== undefined && previousPeriod && (
            <p className="text-xs text-muted-foreground">
              {previousValue} in {previousPeriod}
            </p>
          )}
        </div>
        
        {description && (
          <p className="text-xs text-muted-foreground mt-1">
            {description}
          </p>
        )}
      </div>
    </>
  )

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <motion.div 
            className={cn(metricCardVariants({ variant, isLoading }), 
              onClick ? 'cursor-pointer transition-transform hover:scale-[1.02]' : '',
              className
            )}
            onClick={onClick}
            initial={animate ? { opacity: 0, y: 20 } : {}}
            animate={animate ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5 }}
          >
            {cardContent}
          </motion.div>
        </TooltipTrigger>
        {tooltip && (
          <TooltipContent>
            <p>{tooltip}</p>
          </TooltipContent>
        )}
      </Tooltip>
    </TooltipProvider>
  )
}

// Export common icons for convenience
export const MetricIcons = {
  BarChart: (props: React.SVGProps<SVGSVGElement>) => <BarChart3Icon className="h-4 w-4" {...props} />,
  Percentage: (props: React.SVGProps<SVGSVGElement>) => <Percent className="h-4 w-4" {...props} />,
  Time: (props: React.SVGProps<SVGSVGElement>) => <Clock className="h-4 w-4" {...props} />,
  Alert: (props: React.SVGProps<SVGSVGElement>) => <AlertCircle className="h-4 w-4" {...props} />,
  Success: (props: React.SVGProps<SVGSVGElement>) => <CheckCircle2 className="h-4 w-4" {...props} />,
} 