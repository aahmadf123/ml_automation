"use client"

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { HelpCircle } from 'lucide-react'
import { GuidedTour, TourStep } from '@/components/ui/guided-tour'

// Define the tour steps for the guided application tour
const tourSteps: TourStep[] = [
  {
    target: '.dashboard-overview',
    title: 'Dashboard Overview',
    content: 'Welcome to the ML Automation Dashboard! This is where you can monitor model performance and data quality.',
    placement: 'center',
  },
  {
    target: '.model-comparison',
    title: 'Model Comparison',
    content: 'Compare Model1 (Traditional) and Model4 (Enhanced) performance metrics side by side.',
    placement: 'bottom',
  },
  {
    target: '.business-impact',
    title: 'Business Impact',
    content: 'See how improved model performance translates to real business value and competitive advantages.',
    placement: 'left',
  },
  {
    target: '.drift-monitoring',
    title: 'Drift Monitoring',
    content: 'Track changes in model performance and data distribution over time to ensure reliability.',
    placement: 'right',
  },
  {
    target: '.data-quality',
    title: 'Data Quality',
    content: 'Monitor data quality metrics and anomalies to ensure accurate predictions.',
    placement: 'top',
  },
]

interface GuidedTourButtonProps {
  className?: string
  variant?: 'default' | 'outline' | 'secondary' | 'ghost' | 'link' | 'destructive'
  size?: 'default' | 'sm' | 'lg' | 'icon'
  showIcon?: boolean
  label?: string
}

export function GuidedTourButton({
  className,
  variant = 'outline',
  size = 'default',
  showIcon = true,
  label = 'Take a Tour',
}: GuidedTourButtonProps) {
  const [isTourOpen, setIsTourOpen] = useState(false)

  const startTour = () => {
    setIsTourOpen(true)
  }

  const closeTour = () => {
    setIsTourOpen(false)
  }

  return (
    <>
      <Button
        onClick={startTour}
        variant={variant}
        size={size}
        className={className}
        aria-label="Start guided tour"
      >
        {showIcon && <HelpCircle className="h-4 w-4 mr-2" />}
        {label}
      </Button>

      {isTourOpen && (
        <GuidedTour
          steps={tourSteps}
          isOpen={isTourOpen}
          onClose={closeTour}
          onRequestClose={closeTour}
        />
      )}
    </>
  )
}

