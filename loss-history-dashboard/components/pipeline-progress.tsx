"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { CheckCircle, Clock, Play, RefreshCw, AlertCircle, ArrowRight, ArrowUpRight, ArrowDownRight, LineChart } from "lucide-react"
import { Button } from "@/components/ui/button"
import { motion, AnimatePresence } from "framer-motion"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

type PipelineStage = {
  id: string
  name: string
  status: "pending" | "in-progress" | "completed" | "failed"
  progress: number
  startTime?: Date
  endTime?: Date
  metrics?: {
    dataProcessed?: number
    errorRate?: number
    duration?: number
    performanceScore?: number
    trend?: "up" | "down" | "stable"
  }
}

export function PipelineProgress() {
  const [stages, setStages] = useState<PipelineStage[]>([
    { 
      id: "data-ingestion", 
      name: "Data Ingestion", 
      status: "pending", 
      progress: 0,
      metrics: {
        dataProcessed: 0,
        errorRate: 0.05,
        duration: 0,
        performanceScore: 92,
        trend: "stable"
      }
    },
    { 
      id: "preprocessing", 
      name: "Preprocessing", 
      status: "pending", 
      progress: 0,
      metrics: {
        dataProcessed: 0,
        errorRate: 0.03,
        duration: 0,
        performanceScore: 95,
        trend: "up"
      }
    },
    { 
      id: "feature-engineering", 
      name: "Feature Engineering", 
      status: "pending", 
      progress: 0,
      metrics: {
        dataProcessed: 0,
        errorRate: 0.07,
        duration: 0,
        performanceScore: 88,
        trend: "up"
      }
    },
    { 
      id: "model-training", 
      name: "Model Training", 
      status: "pending", 
      progress: 0,
      metrics: {
        dataProcessed: 0,
        errorRate: 0.04,
        duration: 0,
        performanceScore: 91,
        trend: "up"
      }
    },
    { 
      id: "evaluation", 
      name: "Evaluation", 
      status: "pending", 
      progress: 0,
      metrics: {
        dataProcessed: 0,
        errorRate: 0.02,
        duration: 0,
        performanceScore: 96,
        trend: "up"
      }
    },
    { 
      id: "deployment", 
      name: "Deployment", 
      status: "pending", 
      progress: 0,
      metrics: {
        dataProcessed: 0,
        errorRate: 0.01,
        duration: 0,
        performanceScore: 98,
        trend: "stable"
      }
    },
  ])
  
  const [currentStageIndex, setCurrentStageIndex] = useState<number>(0)
  const [isRunning, setIsRunning] = useState<boolean>(false)
  const [overallProgress, setOverallProgress] = useState<number>(0)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [selectedStage, setSelectedStage] = useState<PipelineStage | null>(null)
  const [isDialogOpen, setIsDialogOpen] = useState<boolean>(false)
  const [hasHistoricalData, setHasHistoricalData] = useState<boolean>(false)
  
  // Reference to store interval IDs
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  
  // Simulate pipeline progress
  useEffect(() => {
    if (!isRunning) return
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
    
    intervalRef.current = setInterval(() => {
      setStages((prevStages) => {
        const newStages = [...prevStages]
        const currentStage = newStages[currentStageIndex]
        
        if (currentStage.status === "pending") {
          currentStage.status = "in-progress"
          currentStage.startTime = new Date()
        }
        
        if (currentStage.progress < 100) {
          // Varied progress increment based on stage type
          let increment = 1
          
          // Different stages progress at different speeds
          switch (currentStage.id) {
            case "data-ingestion":
              increment = Math.floor(Math.random() * 4) + 2 // Faster
              break
            case "model-training":
              increment = Math.floor(Math.random() * 2) + 1 // Slower
              break
            case "deployment":
              increment = Math.floor(Math.random() * 5) + 3 // Faster
              break
            default:
              increment = Math.floor(Math.random() * 3) + 2 // Medium
          }
          
          // Occasionally simulate a brief slowdown
          if (Math.random() < 0.1) {
            increment = 1
          }
          
          currentStage.progress = Math.min(100, currentStage.progress + increment)
          
          // Update metrics as the stage progresses
          if (currentStage.metrics) {
            currentStage.metrics.dataProcessed = Math.floor(currentStage.progress * 12.5)
            currentStage.metrics.duration = Math.floor(currentStage.progress * 0.6)
            
            // Randomly adjust error rate and performance score slightly
            if (Math.random() < 0.2) {
              const errorDelta = (Math.random() - 0.5) * 0.01
              currentStage.metrics.errorRate = Math.max(0, Math.min(0.1, (currentStage.metrics.errorRate || 0) + errorDelta))
              
              const perfDelta = Math.floor((Math.random() - 0.3) * 2)
              currentStage.metrics.performanceScore = Math.max(80, Math.min(100, (currentStage.metrics.performanceScore || 90) + perfDelta))
              
              // Update trend
              if (perfDelta > 0) {
                currentStage.metrics.trend = "up"
              } else if (perfDelta < 0) {
                currentStage.metrics.trend = "down"
              } else {
                currentStage.metrics.trend = "stable"
              }
            }
          }
          
          setLastUpdate(new Date())
        } else if (currentStage.status === "in-progress") {
          currentStage.status = "completed"
          currentStage.endTime = new Date()
          setHasHistoricalData(true)
          
          // Move to next stage
          if (currentStageIndex < stages.length - 1) {
            setCurrentStageIndex(currentStageIndex + 1)
          } else {
            setIsRunning(false)
          }
        }
        
        // Calculate overall progress
        const totalProgress = newStages.reduce((acc, stage) => acc + stage.progress, 0)
        setOverallProgress(Math.floor(totalProgress / newStages.length))
        
        return newStages
      })
    }, 800)
    
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [isRunning, currentStageIndex, stages.length])
  
  // Simulate starting the pipeline
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsRunning(true)
    }, 1500)
    
    return () => clearTimeout(timer)
  }, [])
  
  // Restart the pipeline simulation
  const restartPipeline = () => {
    setStages((prevStages) =>
      prevStages.map((stage) => ({
        ...stage,
        status: "pending",
        progress: 0,
        startTime: undefined,
        endTime: undefined,
        metrics: {
          ...stage.metrics,
          dataProcessed: 0,
          duration: 0,
        }
      }))
    )
    setCurrentStageIndex(0)
    setOverallProgress(0)
    setIsRunning(true)
    setLastUpdate(new Date())
  }

  // Pause/resume pipeline
  const togglePipeline = () => {
    setIsRunning(prev => !prev)
  }
  
  // Open stage details dialog
  const openStageDetails = (stage: PipelineStage) => {
    setSelectedStage(stage)
    setIsDialogOpen(true)
  }
  
  // Get status color for step
  const getStatusColor = (status: PipelineStage["status"]) => {
    switch (status) {
      case "pending":
        return "text-gray-400 border-gray-200"
      case "in-progress":
        return "text-blue-500 border-blue-500"
      case "completed":
        return "text-green-500 border-green-500 bg-green-50"
      case "failed":
        return "text-red-500 border-red-500 bg-red-50"
      default:
        return "text-gray-400 border-gray-200"
    }
  }
  
  // Get status icon
  const getStatusIcon = (status: PipelineStage["status"]) => {
    switch (status) {
      case "pending":
        return <Clock className="h-4 w-4" />
      case "in-progress":
        return <Play className="h-4 w-4" />
      case "completed":
        return <CheckCircle className="h-4 w-4" />
      case "failed":
        return <AlertCircle className="h-4 w-4" />
      default:
        return null
    }
  }

  // Get trend icon
  const getTrendIcon = (trend?: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return <ArrowUpRight className="h-3 w-3 text-green-500" />
      case "down":
        return <ArrowDownRight className="h-3 w-3 text-red-500" />
      default:
        return <ArrowRight className="h-3 w-3 text-gray-400" />
    }
  }
  
  return (
    <>
      <Card className="w-full max-w-full">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center text-lg">
                Pipeline Progress
                {isRunning && (
                  <div className="ml-2 flex items-center">
                    <div className="h-2 w-2 rounded-full bg-green-500 mr-1 animate-pulse" />
                    <span className="text-xs text-green-600 font-normal">LIVE</span>
                  </div>
                )}
              </CardTitle>
              <CardDescription>ML pipeline execution stages (click for details)</CardDescription>
            </div>
            <div className="flex space-x-2">
              <Button 
                onClick={togglePipeline}
                variant="outline"
                size="sm"
                className="h-7 px-2 text-xs"
              >
                {isRunning ? "Pause" : "Resume"}
              </Button>
              <Button 
                onClick={restartPipeline}
                variant="outline"
                size="sm"
                className="h-7 w-7 p-0"
              >
                <RefreshCw className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pb-4 px-2">
          {/* Horizontal Stepper */}
          <div className="w-full flex justify-center py-2">
            <div className="relative flex items-center w-full max-w-[900px]">
              {stages.map((stage, index) => (
                <div 
                  key={stage.id} 
                  className="flex flex-col items-center relative"
                  style={{ 
                    width: `${100 / stages.length}%`,
                    zIndex: 10
                  }}
                >
                  {/* Step Circle */}
                  <div className="relative mb-1">
                    <motion.div
                      className={`flex h-8 w-8 md:h-10 md:w-10 items-center justify-center rounded-full border-2 ${getStatusColor(stage.status)} ${stage.progress > 0 ? 'cursor-pointer hover:shadow-md transition-shadow' : ''}`}
                      animate={{
                        scale: stage.status === 'in-progress' ? [1, 1.1, 1] : 1,
                        transition: {
                          repeat: stage.status === 'in-progress' ? Infinity : 0,
                          duration: 2
                        }
                      }}
                      onClick={() => stage.progress > 0 && openStageDetails(stage)}
                      whileHover={stage.progress > 0 ? { scale: 1.05 } : {}}
                    >
                      {getStatusIcon(stage.status)}
                      <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-white text-[10px] font-medium border">
                        {index + 1}
                      </span>
                    </motion.div>
                    
                    {/* Progress indicator for active step */}
                    {stage.status === 'in-progress' && stage.progress < 100 && (
                      <svg viewBox="0 0 36 36" className="absolute -top-1 -left-1 h-10 w-10 rotate-[-90deg]">
                        <circle 
                          cx="18" 
                          cy="18" 
                          r="16" 
                          fill="none" 
                          className="stroke-blue-100" 
                          strokeWidth="3" 
                        />
                        <circle 
                          cx="18" 
                          cy="18" 
                          r="16" 
                          fill="none" 
                          className="stroke-blue-500" 
                          strokeWidth="3" 
                          strokeDasharray={`${stage.progress}, 100`} 
                        />
                      </svg>
                    )}
                  </div>
                  
                  {/* Step Label */}
                  <div className="text-center">
                    <p className="text-[10px] md:text-xs font-medium whitespace-nowrap overflow-hidden text-ellipsis max-w-full px-1">{stage.name}</p>
                    {stage.progress > 0 && (
                      <p className="text-[10px] text-gray-500 flex items-center justify-center gap-0.5">
                        {stage.progress}%
                        {stage.metrics?.trend && getTrendIcon(stage.metrics.trend)}
                      </p>
                    )}
                  </div>
                </div>
              ))}
              
              {/* Connector Lines */}
              <div className="absolute top-4 left-0 right-0 flex items-center w-full">
                {stages.map((_, index) => (
                  index < stages.length - 1 && (
                    <div 
                      key={`connector-${index}`} 
                      className={`h-0.5 ${index < currentStageIndex ? 'bg-green-500' : 'bg-gray-200'}`}
                      style={{ width: `${100 / (stages.length - 1)}%` }}
                    />
                  )
                ))}
              </div>
            </div>
          </div>
          
          {/* Timeline for current stage progress */}
          {isRunning && (
            <div className="mt-3 px-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-xs font-medium">Current: {stages[currentStageIndex].name}</span>
                <span className="text-[10px] text-gray-500">
                  Updated: {lastUpdate.toLocaleTimeString()}
                </span>
              </div>
              
              <div className="h-1.5 w-full bg-gray-100 rounded-full mt-1 overflow-hidden">
                <motion.div 
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${stages[currentStageIndex].progress}%` }}
                  initial={{ width: 0 }}
                  animate={{ width: `${stages[currentStageIndex].progress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          )}
        </CardContent>
        {hasHistoricalData && (
          <CardFooter className="pt-0 pb-3 px-4">
            <Button variant="ghost" size="sm" className="text-xs w-full">
              <LineChart className="h-3 w-3 mr-1" />
              View historical pipeline runs
            </Button>
          </CardFooter>
        )}
      </Card>

      {/* Stage Details Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedStage && getStatusIcon(selectedStage.status)}
              {selectedStage?.name} Details
              <Badge variant="outline" className={
                selectedStage?.status === "completed" ? "bg-green-50 text-green-800" :
                selectedStage?.status === "in-progress" ? "bg-blue-50 text-blue-800" :
                selectedStage?.status === "failed" ? "bg-red-50 text-red-800" :
                "bg-gray-100 text-gray-800"
              }>
                {selectedStage?.status}
              </Badge>
            </DialogTitle>
            <DialogDescription>
              Performance and execution metrics for this pipeline stage
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {/* Progress & Time */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Progress</h4>
                <div className="flex justify-between text-sm">
                  <span>{selectedStage?.progress || 0}%</span>
                  <span className="text-gray-500">
                    {selectedStage?.startTime ? 
                      new Date(selectedStage.startTime).toLocaleTimeString() : 'Not started'}
                  </span>
                </div>
                <Progress value={selectedStage?.progress || 0} className="h-2" />
              </div>
              <div>
                <h4 className="text-sm font-medium">Duration</h4>
                <p className="text-2xl font-bold mt-2">
                  {selectedStage?.metrics?.duration || 0}s
                </p>
                {selectedStage?.endTime && (
                  <p className="text-xs text-gray-500">
                    Completed: {new Date(selectedStage.endTime).toLocaleTimeString()}
                  </p>
                )}
              </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-4">
              <Card className="border-blue-100">
                <CardHeader className="pb-2 pt-4">
                  <CardTitle className="text-sm">Performance Score</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-end justify-between">
                    <p className="text-2xl font-bold">
                      {selectedStage?.metrics?.performanceScore || 0}%
                    </p>
                    {selectedStage?.metrics?.trend && (
                      <div className="flex items-center gap-1 text-xs">
                        {getTrendIcon(selectedStage.metrics.trend)}
                        <span className={
                          selectedStage.metrics.trend === "up" ? "text-green-600" :
                          selectedStage.metrics.trend === "down" ? "text-red-600" :
                          "text-gray-600"
                        }>
                          {selectedStage.metrics.trend === "up" ? "+1.2%" :
                           selectedStage.metrics.trend === "down" ? "-0.8%" :
                           "No change"}
                        </span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
              <Card className="border-amber-100">
                <CardHeader className="pb-2 pt-4">
                  <CardTitle className="text-sm">Error Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">
                    {(selectedStage?.metrics?.errorRate || 0) * 100}%
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Detailed Metrics */}
            <Card>
              <CardHeader className="pb-2 pt-4">
                <CardTitle className="text-sm">Processing Details</CardTitle>
              </CardHeader>
              <CardContent className="pb-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Data Processed</span>
                    <span className="text-sm font-medium">
                      {selectedStage?.metrics?.dataProcessed || 0} MB
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Execution Time</span>
                    <span className="text-sm font-medium">
                      {selectedStage?.metrics?.duration || 0} seconds
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Success Rate</span>
                    <span className="text-sm font-medium">
                      {100 - ((selectedStage?.metrics?.errorRate || 0) * 100)}%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                Close
              </Button>
              <Button>
                View Detailed Logs
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
} 