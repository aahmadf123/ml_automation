"use client"

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { 
  ArrowUpRight, 
  ArrowDownRight,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Clock,
  BarChart3,
  Activity
} from 'lucide-react'

interface StatCardProps {
  title: string
  value: string | number
  description?: string
  trend?: number
  icon?: React.ReactNode
  status?: 'success' | 'warning' | 'error' | 'loading' | 'neutral'
  className?: string
}

interface QuickStatsProps {
  stats: StatCardProps[]
  columns?: 2 | 3 | 4
  className?: string
}

export function QuickStats({ 
  stats,
  columns = 4,
  className 
}: QuickStatsProps) {
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
  
  const columnClasses = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  }

  return (
    <div ref={ref} className={cn('w-full', className)}>
      <div className={cn('grid gap-4', columnClasses[columns])}>
        {stats.map((stat, index) => (
          <StatCard 
            key={`stat-${index}`}
            title={stat.title}
            value={stat.value}
            description={stat.description}
            trend={stat.trend}
            icon={stat.icon}
            status={stat.status}
            className={stat.className}
            animate={isVisible}
            delay={index * 0.1}
          />
        ))}
      </div>
    </div>
  )
}

function StatCard({
  title,
  value,
  description,
  trend,
  icon,
  status = 'neutral',
  className,
  animate = true,
  delay = 0
}: StatCardProps & { 
  animate?: boolean
  delay?: number
}) {
  const [animatedValue, setAnimatedValue] = useState(0)
  const isNumeric = typeof value === 'number'
  const displayValue = isNumeric ? animatedValue : value
  
  const statusIcon = {
    success: <CheckCircle2 className="h-5 w-5 text-emerald-500" />,
    warning: <AlertTriangle className="h-5 w-5 text-amber-500" />,
    error: <AlertTriangle className="h-5 w-5 text-red-500" />,
    loading: <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />,
    neutral: null
  }
  
  const variantClasses = {
    success: 'border-emerald-100 bg-gradient-to-b from-white to-emerald-50',
    warning: 'border-amber-100 bg-gradient-to-b from-white to-amber-50',
    error: 'border-red-100 bg-gradient-to-b from-white to-red-50',
    loading: 'border-blue-100 bg-gradient-to-b from-white to-blue-50',
    neutral: 'border-slate-200 bg-gradient-to-b from-white to-slate-50'
  }

  useEffect(() => {
    if (!animate || !isNumeric) return
    
    const targetValue = value as number
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
  }, [animate, value, isNumeric])

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
      animate={animate ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay }}
      className={cn(
        'rounded-xl border p-6 shadow-sm relative overflow-hidden',
        variantClasses[status],
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <h3 className="text-sm font-medium text-slate-500">{title}</h3>
          <div className="flex items-baseline space-x-1">
            <span className="text-2xl font-bold text-slate-900">
              {formatValue(displayValue)}
            </span>
            {trend !== undefined && (
              <span 
                className={cn(
                  'text-sm font-medium',
                  trend > 0 ? 'text-emerald-600' : trend < 0 ? 'text-red-600' : 'text-slate-600'
                )}
              >
                {trend > 0 ? (
                  <span className="flex items-center">
                    <ArrowUpRight className="h-3 w-3 mr-0.5" />
                    {Math.abs(trend)}%
                  </span>
                ) : trend < 0 ? (
                  <span className="flex items-center">
                    <ArrowDownRight className="h-3 w-3 mr-0.5" />
                    {Math.abs(trend)}%
                  </span>
                ) : null}
              </span>
            )}
          </div>
          {description && (
            <p className="text-sm text-slate-500 mt-1">{description}</p>
          )}
        </div>
        <div>
          {icon || statusIcon[status]}
        </div>
      </div>
      
      {/* Decorative background element */}
      <div 
        className="absolute -bottom-6 -right-6 h-24 w-24 rounded-full opacity-10 bg-current"
        style={{ color: status === 'neutral' ? '#64748b' : undefined }}
      />
    </motion.div>
  )
} 