"use client"

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { BrainCircuit } from "lucide-react"
import { ChatBox } from "@/components/ui/ChatBox"

interface AIAssistantButtonProps {
  className?: string
  variant?: "default" | "outline" | "secondary" | "ghost" | "link" | "destructive"
  size?: "default" | "sm" | "lg" | "icon"
  buttonText?: string
}

export function AIAssistantButton({
  className,
  variant = "default",
  size = "default",
  buttonText = "AI Assistant",
}: AIAssistantButtonProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return

    setIsLoading(true)
    
    try {
      // In a real implementation, this would call your API
      // Example: const response = await fetch('/api/ai-assistant', { method: 'POST', body: JSON.stringify({ message }) })
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Return mock response
      return {
        id: Date.now().toString(),
        content: `This is a mock response to: "${message}". In a production environment, this would connect to your OpenAI-powered assistant API.`,
        role: 'assistant',
        timestamp: new Date().toISOString()
      }
    } catch (error) {
      console.error('Error sending message to AI Assistant:', error)
      return {
        id: Date.now().toString(),
        content: "Sorry, I encountered an error processing your request. Please try again.",
        role: 'assistant',
        timestamp: new Date().toISOString()
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      <Button
        variant={variant}
        size={size}
        className={`relative overflow-hidden group ${className}`}
        onClick={() => setIsOpen(true)}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 opacity-0 group-hover:opacity-20 transition-opacity duration-300" />
        
        <BrainCircuit className="mr-2 h-4 w-4" />
        <span>{buttonText}</span>
        
        <span className="absolute top-0 right-0 h-2 w-2 rounded-full bg-green-500 animate-pulse" />
      </Button>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="sm:max-w-[600px] h-[600px] flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center">
              <BrainCircuit className="mr-2 h-5 w-5 text-primary" />
              AI Assistant
            </DialogTitle>
            <DialogDescription>
              Ask questions about your models, data, or get help with analysis
            </DialogDescription>
          </DialogHeader>
          
          <div className="flex-1 overflow-hidden">
            <ChatBox
              title="Model Assistant"
              description="I can help you analyze model performance, explain predictions, and provide insights on your data."
              placeholder="Ask me anything about your models or data..."
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              initialMessages={[
                {
                  id: '1',
                  content: "Hello! I'm your AI assistant for model analysis and interpretability. How can I help you today?",
                  role: 'assistant',
                  timestamp: new Date().toISOString()
                }
              ]}
            />
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
} 