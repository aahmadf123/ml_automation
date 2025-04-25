"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { motion } from "framer-motion"
import { AlertCircle, CheckCircle, Clock, Play, RefreshCw, Pulse, Activity } from "lucide-react"
import { Button } from "@/components/ui/button"

type PipelineStage = {
  id: string
  name: string
  status: "pending" | "in-progress" | "completed" | "failed"
  progress: number
  startTime?: Date
  endTime?: Date
}

export function PipelineProgress() {
  const [stages, setStages] = useState<PipelineStage[]>([
    { id: "data-ingestion", name: "Data Ingestion", status: "pending", progress: 0 },
    { id: "preprocessing", name: "Preprocessing", status: "pending", progress: 0 },
    { id: "feature-engineering", name: "Feature Engineering", status: "pending", progress: 0 },
    { id: "model-training", name: "Model Training", status: "pending", progress: 0 },
    { id: "evaluation", name: "Evaluation", status: "pending", progress: 0 },
    { id: "deployment", name: "Deployment", status: "pending", progress: 0 },
  ])
  
  const [currentStageIndex, setCurrentStageIndex] = useState<number>(0)
  const [isRunning, setIsRunning] = useState<boolean>(false)
  const [overallProgress, setOverallProgress] = useState<number>(0)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [pulseDot, setPulseDot] = useState<boolean>(true)
  
  // Reference to store interval IDs
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const pulseIntervalRef = useRef<NodeJS.Timeout | null>(null)
  
  // Function to simulate processing speed in KB/s
  const [processingSpeed, setProcessingSpeed] = useState<number>(0)
  
  // Calculate processing speed
  useEffect(() => {
    if (isRunning) {
      const updateSpeed = () => {
        // Random speed between 400-800 KB/s
        setProcessingSpeed(Math.floor(Math.random() * 400) + 400)
      }
      
      updateSpeed() // Initial call
      const speedInterval = setInterval(updateSpeed, 3000)
      
      return () => clearInterval(speedInterval)
    } else {
      setProcessingSpeed(0)
    }
  }, [isRunning])
  
  // Simulate pipeline progress with more realistic behavior
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
          setLastUpdate(new Date())
        } else if (currentStage.status === "in-progress") {
          currentStage.status = "completed"
          currentStage.endTime = new Date()
          
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
  
  // Pulse animation for "live" indicator
  useEffect(() => {
    if (isRunning) {
      if (pulseIntervalRef.current) {
        clearInterval(pulseIntervalRef.current)
      }
      
      pulseIntervalRef.current = setInterval(() => {
        setPulseDot(prev => !prev)
      }, 1000)
    }
    
    return () => {
      if (pulseIntervalRef.current) clearInterval(pulseIntervalRef.current)
    }
  }, [isRunning])
  
  // Restart the pipeline simulation
  const restartPipeline = () => {
    setStages((prevStages) =>
      prevStages.map((stage) => ({
        ...stage,
        status: "pending",
        progress: 0,
        startTime: undefined,
        endTime: undefined,
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
  
  // Status badge component
  const StatusBadge = ({ status }: { status: PipelineStage["status"] }) => {
    switch (status) {
      case "pending":
        return <Badge variant="outline" className="flex items-center gap-1 text-xs"><Clock className="h-3 w-3" /> Pending</Badge>
      case "in-progress":
        return <Badge variant="secondary" className="flex items-center gap-1 text-xs bg-blue-100 text-blue-800"><Play className="h-3 w-3" /> Running</Badge>
      case "completed":
        return <Badge variant="default" className="flex items-center gap-1 text-xs bg-green-100 text-green-800"><CheckCircle className="h-3 w-3" /> Completed</Badge>
      case "failed":
        return <Badge variant="destructive" className="flex items-center gap-1 text-xs"><AlertCircle className="h-3 w-3" /> Failed</Badge>
      default:
        return null
    }
  }
  
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center">
              Pipeline Progress
              {isRunning && (
                <div className="ml-2 flex items-center">
                  <div className={`h-2 w-2 rounded-full ${pulseDot ? 'bg-green-500' : 'bg-green-300'} mr-1`} />
                  <span className="text-xs text-green-600 font-normal">LIVE</span>
                </div>
              )}
            </CardTitle>
            <CardDescription>Real-time ML pipeline execution</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={isRunning ? "default" : "outline"} className="h-6">
              {isRunning ? "Running" : "Completed"}
            </Badge>
            <div className="flex space-x-1">
              <Button 
                onClick={togglePipeline}
                variant="outline"
                size="sm"
                className="h-8 px-2"
              >
                {isRunning ? "Pause" : "Resume"}
              </Button>
              <button 
                onClick={restartPipeline}
                className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-medium">Overall Progress</div>
            <div className="text-sm font-medium">{overallProgress}%</div>
          </div>
          <Progress value={overallProgress} className="h-2" />
          
          {/* Live Stats */}
          <div className="flex justify-between mt-2 text-xs text-gray-500">
            <div className="flex items-center">
              <Activity className="h-3 w-3 mr-1" />
              <span>{processingSpeed} KB/s</span>
            </div>
            <div>
              Last updated: {lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </div>
        
        <div className="space-y-4">
          {stages.map((stage, index) => (
            <motion.div 
              key={stage.id}
              className={`rounded-lg border p-3 ${
                stage.status === "in-progress" 
                  ? "border-blue-200 bg-blue-50" 
                  : stage.status === "completed" 
                    ? "border-green-200 bg-green-50" 
                    : "border-gray-200"
              }`}
              initial={{ opacity: 0.6, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="font-medium">{stage.name}</div>
                <StatusBadge status={stage.status} />
              </div>
              <Progress value={stage.progress} className="h-1.5 mb-2" />
              <div className="flex items-center justify-between text-xs text-gray-500">
                <div>
                  {stage.startTime ? new Date(stage.startTime).toLocaleTimeString() : '--:--:--'}
                </div>
                <div>{stage.progress}%</div>
                <div>
                  {stage.endTime ? new Date(stage.endTime).toLocaleTimeString() : '--:--:--'}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
} 