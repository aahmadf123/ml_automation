"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Card, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { X, Play, Save, Folder, ChevronDown, ChevronUp } from "lucide-react"

interface CodeInterpreterProps {
  onClose: () => void
}

export function CodeInterpreter({ onClose }: CodeInterpreterProps) {
  const [code, setCode] = useState("# Enter your Python code here\n\n")
  const [output, setOutput] = useState("")
  const [isRunning, setIsRunning] = useState(false)
  const [isFileExplorerOpen, setIsFileExplorerOpen] = useState(false)
  const [history, setHistory] = useState<string[]>([])
  const dragRef = useRef<HTMLDivElement>(null)
  const [position, setPosition] = useState({ x: 100, y: 100 })
  const [size, setSize] = useState({ width: 800, height: 500 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [isResizing, setIsResizing] = useState(false)
  const [resizeDirection, setResizeDirection] = useState<string | null>(null)

  // Handle dragging
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        setPosition({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y,
        })
      }

      if (isResizing) {
        if (resizeDirection === "se") {
          setSize({
            width: Math.max(400, e.clientX - position.x),
            height: Math.max(300, e.clientY - position.y),
          })
        }
      }
    }

    const handleMouseUp = () => {
      setIsDragging(false)
      setIsResizing(false)
      setResizeDirection(null)
    }

    document.addEventListener("mousemove", handleMouseMove)
    document.addEventListener("mouseup", handleMouseUp)

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isDragging, isResizing, dragOffset, position, resizeDirection])

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true)
    setDragOffset({
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    })
  }

  const handleResizeMouseDown = (e: React.MouseEvent<HTMLDivElement>, direction: string) => {
    e.stopPropagation()
    setIsResizing(true)
    setResizeDirection(direction)
  }

  const runCode = () => {
    setIsRunning(true)
    setOutput("Running code...\n")

    // Simulate code execution
    setTimeout(() => {
      setOutput((prev) => prev + "Code execution completed.\n\n# Output would appear here\n")
      setIsRunning(false)
      setHistory((prev) => [...prev, code])
    }, 1500)
  }

  return (
    <div
      className="fixed z-50"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        width: `${size.width}px`,
        height: `${size.height}px`,
      }}
    >
      <Card className="h-full flex flex-col shadow-xl border-primary/20">
        <CardHeader
          className="py-2 px-4 cursor-move bg-muted/50 border-b flex flex-row items-center justify-between"
          ref={dragRef}
          onMouseDown={handleMouseDown}
        >
          <CardTitle className="text-sm font-medium">Code Interpreter Console</CardTitle>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>

        <div className="flex flex-1 overflow-hidden">
          {isFileExplorerOpen && (
            <div className="w-48 border-r p-2 overflow-y-auto">
              <div className="font-medium text-xs mb-2 text-muted-foreground">FILE EXPLORER</div>
              <div className="space-y-1">
                <div className="flex items-center text-xs py-1 px-2 rounded hover:bg-muted cursor-pointer">
                  <Folder className="h-3 w-3 mr-2" />
                  <span>/tmp</span>
                </div>
                <div className="flex items-center text-xs py-1 px-2 rounded hover:bg-muted cursor-pointer pl-6">
                  <span>data.csv</span>
                </div>
                <div className="flex items-center text-xs py-1 px-2 rounded hover:bg-muted cursor-pointer pl-6">
                  <span>results.json</span>
                </div>
                <div className="flex items-center text-xs py-1 px-2 rounded hover:bg-muted cursor-pointer pl-6">
                  <span>plot.png</span>
                </div>
              </div>
            </div>
          )}

          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="flex border-b">
              <Button
                variant="ghost"
                size="sm"
                className="rounded-none h-8 text-xs"
                onClick={() => setIsFileExplorerOpen(!isFileExplorerOpen)}
              >
                {isFileExplorerOpen ? <ChevronUp className="h-3 w-3 mr-1" /> : <ChevronDown className="h-3 w-3 mr-1" />}
                Files
              </Button>
              <Button variant="ghost" size="sm" className="rounded-none h-8 text-xs">
                <Save className="h-3 w-3 mr-1" />
                Save
              </Button>
            </div>

            <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
              <div className="flex-1 p-2 overflow-auto">
                <textarea
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  className="w-full h-full p-2 font-mono text-sm bg-muted/30 rounded border-0 focus:ring-0 resize-none"
                  spellCheck="false"
                />
              </div>

              <div className="flex-1 border-t md:border-t-0 md:border-l overflow-hidden flex flex-col">
                <div className="p-2 flex items-center justify-between border-b">
                  <span className="text-xs font-medium">Output</span>
                  <Button size="sm" className="h-7 text-xs" onClick={runCode} disabled={isRunning}>
                    <Play className="h-3 w-3 mr-1" />
                    Run
                  </Button>
                </div>
                <div className="flex-1 p-2 overflow-auto">
                  <pre className="text-xs font-mono whitespace-pre-wrap">{output}</pre>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="border-t p-2 bg-muted/30">
          <div className="text-xs font-medium mb-1">History</div>
          <div className="max-h-20 overflow-y-auto">
            {history.length === 0 ? (
              <div className="text-xs text-muted-foreground">No history yet</div>
            ) : (
              history.map((item, index) => (
                <div
                  key={index}
                  className="text-xs py-1 px-2 rounded hover:bg-muted cursor-pointer truncate"
                  onClick={() => setCode(item)}
                >
                  {item.split("\n")[0]}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Resize handle */}
        <div
          className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize"
          onMouseDown={(e) => handleResizeMouseDown(e, "se")}
        >
          <div className="w-2 h-2 border-r-2 border-b-2 border-foreground/30 absolute bottom-1 right-1" />
        </div>
      </Card>
    </div>
  )
}
