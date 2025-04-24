"use client"

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, ChevronRight, ChevronLeft, Info, Lightbulb, Key, Activity } from 'lucide-react'
import { Button } from './button'
import { cn } from '@/lib/utils'

type Step = {
  id: string
  title: string
  content: React.ReactNode
  target?: string // CSS selector for the element to highlight
  placement?: 'top' | 'bottom' | 'left' | 'right'
  icon?: React.ReactNode
}

export interface GuidedTourProps {
  steps: Step[]
  onComplete?: () => void
  onSkip?: () => void
  isOpen?: boolean
  className?: string
  spotlightClassName?: string
  onOpenChange?: (open: boolean) => void
  defaultOpen?: boolean
  tourName?: string // Used for storing in localStorage
}

export function GuidedTour({
  steps,
  onComplete,
  onSkip,
  isOpen: controlledIsOpen,
  className,
  spotlightClassName,
  onOpenChange,
  defaultOpen = false,
  tourName = 'default-tour',
}: GuidedTourProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isOpen, setIsOpen] = useState(defaultOpen)
  const [targetElement, setTargetElement] = useState<Element | null>(null)
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null)
  
  // Handle controlled/uncontrolled component pattern
  const open = controlledIsOpen !== undefined ? controlledIsOpen : isOpen
  
  const localStorageKey = `guided-tour-${tourName}-completed`
  
  // Check if tour has been completed before
  useEffect(() => {
    const hasCompleted = localStorage.getItem(localStorageKey)
    if (!hasCompleted && defaultOpen) {
      setIsOpen(true)
      if (onOpenChange) onOpenChange(true)
    }
  }, [defaultOpen, localStorageKey, onOpenChange])
  
  // Find and track target element position
  useEffect(() => {
    if (!open) return
    
    const target = steps[currentStep]?.target
    if (!target) {
      setTargetElement(null)
      setTargetRect(null)
      return
    }
    
    const element = document.querySelector(target)
    setTargetElement(element)
    
    if (element) {
      setTargetRect(element.getBoundingClientRect())
      
      // Scroll element into view if needed
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      })
      
      // Add highlight class to target element
      element.classList.add('guided-tour-target')
    }
    
    const handleResize = () => {
      if (element) {
        setTargetRect(element.getBoundingClientRect())
      }
    }
    
    window.addEventListener('resize', handleResize)
    
    return () => {
      window.removeEventListener('resize', handleResize)
      // Remove highlight class
      if (element) {
        element.classList.remove('guided-tour-target')
      }
    }
  }, [open, currentStep, steps])
  
  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      handleComplete()
    }
  }
  
  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }
  
  const handleComplete = () => {
    // Mark tour as completed in localStorage
    localStorage.setItem(localStorageKey, 'true')
    
    // Close the tour
    setIsOpen(false)
    if (onOpenChange) onOpenChange(false)
    
    // Call onComplete callback
    if (onComplete) onComplete()
    
    // Reset to first step for next time
    setCurrentStep(0)
  }
  
  const handleSkip = () => {
    // Mark tour as completed in localStorage
    localStorage.setItem(localStorageKey, 'true')
    
    // Close the tour
    setIsOpen(false)
    if (onOpenChange) onOpenChange(false)
    
    // Call onSkip callback
    if (onSkip) onSkip()
    
    // Reset to first step for next time
    setCurrentStep(0)
  }
  
  // If not open, don't render anything
  if (!open) return null
  
  const currentStepData = steps[currentStep]
  const isLastStep = currentStep === steps.length - 1
  
  // Calculate tooltip position based on target element
  const getTooltipPosition = () => {
    if (!targetRect) {
      return {
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
      }
    }
    
    const placement = currentStepData.placement || 'bottom'
    const padding = 10 // Padding from the target element
    
    switch (placement) {
      case 'top':
        return {
          bottom: `${window.innerHeight - targetRect.top + padding}px`,
          left: `${targetRect.left + targetRect.width / 2}px`,
          transform: 'translateX(-50%)',
        }
      case 'bottom':
        return {
          top: `${targetRect.bottom + padding}px`,
          left: `${targetRect.left + targetRect.width / 2}px`,
          transform: 'translateX(-50%)',
        }
      case 'left':
        return {
          top: `${targetRect.top + targetRect.height / 2}px`,
          right: `${window.innerWidth - targetRect.left + padding}px`,
          transform: 'translateY(-50%)',
        }
      case 'right':
        return {
          top: `${targetRect.top + targetRect.height / 2}px`,
          left: `${targetRect.right + padding}px`,
          transform: 'translateY(-50%)',
        }
      default:
        return {
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
        }
    }
  }
  
  // For spotlight effect
  const getSpotlightStyle = () => {
    if (!targetRect) {
      return {}
    }
    
    return {
      top: `${targetRect.top}px`,
      left: `${targetRect.left}px`,
      width: `${targetRect.width}px`,
      height: `${targetRect.height}px`,
    }
  }
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <motion.div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={handleSkip}
      />
      
      {/* Spotlight effect */}
      {targetRect && (
        <motion.div
          className={cn(
            "absolute rounded-md transition-all duration-300 box-content border-2 border-primary z-[51]",
            spotlightClassName
          )}
          style={getSpotlightStyle()}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
        />
      )}
      
      {/* Tooltip */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          className={cn(
            "fixed z-[52] w-80 bg-card text-card-foreground rounded-lg shadow-lg",
            className
          )}
          style={getTooltipPosition()}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center gap-2">
              {currentStepData.icon || <Info className="w-5 h-5 text-primary" />}
              <h3 className="font-medium">{currentStepData.title}</h3>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={handleSkip}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          {/* Content */}
          <div className="p-4">
            <div className="mb-4">{currentStepData.content}</div>
            
            {/* Progress indicators */}
            <div className="flex items-center justify-center gap-1 mb-4">
              {steps.map((_, index) => (
                <div
                  key={index}
                  className={cn(
                    "h-1.5 rounded-full transition-all duration-300",
                    index === currentStep 
                      ? "w-4 bg-primary" 
                      : "w-1.5 bg-muted-foreground/30"
                  )}
                />
              ))}
            </div>
            
            {/* Actions */}
            <div className="flex items-center justify-between">
              <Button
                variant="ghost"
                size="sm"
                onClick={handlePrev}
                disabled={currentStep === 0}
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Previous
              </Button>
              
              <Button
                variant="default"
                size="sm"
                onClick={handleNext}
              >
                {isLastStep ? 'Finish' : 'Next'}
                {!isLastStep && <ChevronRight className="h-4 w-4 ml-1" />}
              </Button>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
} 