"use client"

import { ReactNode, useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardHeader, CardContent } from "@/components/ui/card"
import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export interface SummaryCardProps {
  title: string
  value: string | number
  icon?: LucideIcon
  trend?: "positive" | "negative" | "neutral"
  changeValue?: string | number
  changeText?: string
  tooltipText?: string
  suffix?: string
  className?: string
  animation?: boolean
}

export function SummaryCard({
  title,
  value,
  icon: Icon,
  trend = "neutral",
  changeValue,
  changeText = "from last quarter",
  tooltipText,
  suffix = "",
  className,
  animation = true,
}: SummaryCardProps) {
  const [prevValue, setPrevValue] = useState<string | number>(value)
  const [isValueChanged, setIsValueChanged] = useState(false)
  
  // Detect value changes for animation
  useEffect(() => {
    if (value !== prevValue && animation) {
      setIsValueChanged(true)
      const timer = setTimeout(() => {
        setIsValueChanged(false)
        setPrevValue(value)
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [value, prevValue, animation])

  const trendColors = {
    positive: "text-emerald-500 dark:text-emerald-400",
    negative: "text-red-500 dark:text-red-400",
    neutral: "text-blue-500 dark:text-blue-400"
  }
  
  const trendBgColors = {
    positive: "bg-emerald-50 dark:bg-emerald-950",
    negative: "bg-red-50 dark:bg-red-950",
    neutral: "bg-blue-50 dark:bg-blue-950"
  }
  
  const trendBorderColors = {
    positive: "border-emerald-200 dark:border-emerald-900",
    negative: "border-red-200 dark:border-red-900",
    neutral: "border-blue-200 dark:border-blue-900"
  }

  return (
    <Card className={cn("overflow-hidden", 
      {[`border-l-4 ${trendBorderColors[trend]}`]: trend !== "neutral"},
      className
    )}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <TooltipProvider>
          <Tooltip delayDuration={300}>
            <TooltipTrigger asChild>
              <h3 className="text-sm font-medium cursor-help">{title}</h3>
            </TooltipTrigger>
            {tooltipText && (
              <TooltipContent side="top" align="start">
                <p className="max-w-xs text-sm">{tooltipText}</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
        {Icon && <Icon className={cn("h-4 w-4", trendColors[trend])} />}
      </CardHeader>
      <CardContent>
        <AnimatePresence mode="wait">
          <motion.div
            key={`${value}-${isValueChanged ? "animating" : "static"}`}
            initial={animation ? { opacity: 0, y: 10 } : {}}
            animate={animation ? { opacity: 1, y: 0 } : {}}
            exit={animation ? { opacity: 0, y: -10 } : {}}
            transition={{ duration: 0.3 }}
            className={cn("text-2xl font-bold", {
              "text-emerald-600 dark:text-emerald-400": trend === "positive" && isValueChanged,
              "text-red-600 dark:text-red-400": trend === "negative" && isValueChanged
            })}
          >
            {value}{suffix}
          </motion.div>
        </AnimatePresence>
        {changeValue && (
          <div className="mt-1">
            <p className="text-xs text-muted-foreground flex items-center">
              <span className={cn("flex items-center gap-1", trendColors[trend])}>
                {trend === "positive" ? (
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    width="12" 
                    height="12" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                  >
                    <path d="m18 15-6-6-6 6"/>
                  </svg>
                ) : trend === "negative" ? (
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    width="12" 
                    height="12" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                  >
                    <path d="m6 9 6 6 6-6"/>
                  </svg>
                ) : null}
                {changeValue}
              </span>
              {" "}
              {changeText}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

