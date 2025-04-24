"use client"

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent } from "@/components/ui/card"
import { ChevronLeft, ChevronRight, PauseCircle, PlayCircle, AlertTriangle, LineChart, BrainCircuit, Lightbulb, TrendingUp } from "lucide-react"

// Sample insights data (would normally be fetched from an API)
const insights = [
  {
    id: 1,
    text: "Neural Network model shows 7.8% improved F1 score over baseline model across all claim types",
    type: "performance",
    icon: TrendingUp
  },
  {
    id: 2,
    text: "Detected drift in 'property_age' feature - consider retraining Random Forest model",
    type: "drift",
    icon: AlertTriangle
  },
  {
    id: 3,
    text: "Claims with multiple line items show 23% higher processing times - potential optimization opportunity",
    type: "insight",
    icon: Lightbulb
  },
  {
    id: 4,
    text: "Feature importance analysis: 'claim_amount' has increased significance in the latest model version",
    type: "feature",
    icon: LineChart
  },
  {
    id: 5,
    text: "AI Assistant correctly answered 94.2% of customer inquiries without human intervention",
    type: "ai",
    icon: BrainCircuit
  }
];

export function LatestInsightsBanner() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const nextInsight = () => {
    if (isAnimating) return;
    
    setIsAnimating(true);
    setCurrentIndex((prevIndex) => (prevIndex + 1) % insights.length);
    
    setTimeout(() => {
      setIsAnimating(false);
    }, 400);
  };

  const prevInsight = () => {
    if (isAnimating) return;
    
    setIsAnimating(true);
    setCurrentIndex((prevIndex) => 
      prevIndex === 0 ? insights.length - 1 : prevIndex - 1
    );
    
    setTimeout(() => {
      setIsAnimating(false);
    }, 400);
  };

  useEffect(() => {
    if (!isPaused) {
      timerRef.current = setInterval(() => {
        nextInsight();
      }, 5000);
    }
    
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isPaused, isAnimating]);

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'performance':
        return 'text-blue-500';
      case 'drift':
        return 'text-amber-500';
      case 'insight':
        return 'text-purple-500';
      case 'feature':
        return 'text-green-500';
      case 'ai':
        return 'text-indigo-500';
      default:
        return 'text-gray-500';
    }
  };

  const currentInsight = insights[currentIndex];
  const Icon = currentInsight.icon;

  return (
    <Card 
      className="relative overflow-hidden shadow-md border-l-4"
      style={{ borderLeftColor: `var(--${currentInsight.type}-color, var(--primary))` }}
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      <CardContent className="p-4 flex items-center">
        <button 
          onClick={prevInsight} 
          className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="Previous insight"
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        
        <div className={`flex-1 mx-2 transition-opacity duration-300 ${isAnimating ? 'opacity-0' : 'opacity-100'}`}>
          <div className="flex items-center">
            <div className={`mr-3 ${getTypeColor(currentInsight.type)}`}>
              <Icon className="h-5 w-5" />
            </div>
            <p className="text-sm font-medium">{currentInsight.text}</p>
          </div>
        </div>
        
        <div className="flex items-center">
          <button 
            onClick={() => setIsPaused(!isPaused)} 
            className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors mr-1"
            aria-label={isPaused ? "Resume auto-rotation" : "Pause auto-rotation"}
          >
            {isPaused ? (
              <PlayCircle className="h-5 w-5 text-gray-400" />
            ) : (
              <PauseCircle className="h-5 w-5 text-gray-400" />
            )}
          </button>
          
          <button 
            onClick={nextInsight} 
            className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Next insight"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        </div>
        
        <div className="absolute bottom-1 left-0 w-full flex justify-center">
          <div className="flex space-x-1">
            {insights.map((_, index) => (
              <div 
                key={index}
                className={`h-1 rounded-full transition-all duration-300 ${
                  currentIndex === index 
                    ? 'w-4 bg-primary' 
                    : 'w-2 bg-gray-300 dark:bg-gray-700'
                }`}
              />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 