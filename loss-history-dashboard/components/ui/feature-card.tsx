"use client"

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { ChevronRight, ExternalLink } from 'lucide-react'
import { motion } from "framer-motion"

interface FeatureCardProps {
  title: string
  description: string
  icon: React.ReactNode
  className?: string
  link?: string
  linkText?: string
  color?: "default" | "primary" | "secondary" | "accent"
  isExternal?: boolean
  onClick?: () => void
}

export function FeatureCard({
  title,
  description,
  icon,
  className,
  link,
  linkText = "Learn more",
  color = "default",
  isExternal = false,
  onClick
}: FeatureCardProps) {
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

  const getColorStyles = () => {
    switch(color) {
      case "primary":
        return "bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 hover:border-blue-300"
      case "secondary":
        return "bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 hover:border-purple-300"
      case "accent":
        return "bg-gradient-to-br from-amber-50 to-amber-100 border-amber-200 hover:border-amber-300"
      default:
        return "bg-gradient-to-br from-slate-50 to-slate-100 border-slate-200 hover:border-slate-300"
    }
  }

  const iconWrapperStyles = () => {
    switch(color) {
      case "primary":
        return "bg-blue-100 text-blue-700 border-blue-200"
      case "secondary":
        return "bg-purple-100 text-purple-700 border-purple-200"
      case "accent":
        return "bg-amber-100 text-amber-700 border-amber-200"
      default:
        return "bg-slate-100 text-slate-700 border-slate-200"
    }
  }

  const buttonStyles = () => {
    switch(color) {
      case "primary":
        return "bg-blue-100 text-blue-700 hover:bg-blue-200"
      case "secondary":
        return "bg-purple-100 text-purple-700 hover:bg-purple-200"
      case "accent":
        return "bg-amber-100 text-amber-700 hover:bg-amber-200"
      default:
        return "bg-slate-100 text-slate-700 hover:bg-slate-200"
    }
  }

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={{ 
        opacity: isVisible ? 1 : 0, 
        y: isVisible ? 0 : 20 
      }}
      transition={{ duration: 0.5 }}
      className={cn("h-full", className)}
    >
      <Card className={cn(
        "border h-full transition-all duration-300 hover:shadow-md transform hover:-translate-y-1",
        getColorStyles()
      )}>
        <CardHeader className="pb-2">
          <div className={cn(
            "w-12 h-12 rounded-lg flex items-center justify-center mb-3 border",
            iconWrapperStyles()
          )}>
            {icon}
          </div>
          <CardTitle className="text-xl font-bold">{title}</CardTitle>
          <CardDescription className="text-gray-600 mt-2">{description}</CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-gray-500">
          <slot />
        </CardContent>
        {(link || onClick) && (
          <CardFooter className="pt-2">
            <Button 
              variant="ghost" 
              className={cn("px-3 py-2 text-sm flex items-center gap-1", buttonStyles())}
              onClick={onClick}
              asChild={!!link}
            >
              {link ? (
                <a href={link} target={isExternal ? "_blank" : undefined} rel={isExternal ? "noopener noreferrer" : undefined}>
                  {linkText}
                  {isExternal ? <ExternalLink className="ml-1 h-3 w-3" /> : <ChevronRight className="ml-1 h-3 w-3" />}
                </a>
              ) : (
                <>
                  {linkText}
                  <ChevronRight className="ml-1 h-3 w-3" />
                </>
              )}
            </Button>
          </CardFooter>
        )}
      </Card>
    </motion.div>
  )
} 