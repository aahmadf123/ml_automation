'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ChatBox, ChatMessage } from '@/components/ui/ChatBox';
import { Button } from '@/components/ui/button';
import { ArrowLeft, BrainCircuit } from 'lucide-react';
import Link from 'next/link';

export default function AIAssistantPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'Hello! I can help you analyze and understand the model comparison results. What would you like to know?',
      timestamp: new Date()
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (message: string) => {
    // Add user message to chat
    const userMessage: ChatMessage = {
      role: 'user',
      content: message,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Call the AI assistant API
      const response = await fetch('/api/models/ai-assistant', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date(data.timestamp || Date.now()),
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error getting AI response:', error);
      
      // Add error message
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6 flex items-center">
        <Link href="/model-comparison">
          <Button variant="ghost" size="sm" className="gap-1">
            <ArrowLeft className="h-4 w-4" />
            Back to Comparison
          </Button>
        </Link>
        <h1 className="text-3xl font-bold ml-4 flex items-center">
          <BrainCircuit className="mr-2 h-6 w-6" />
          Model Comparison AI Assistant
        </h1>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ChatBox
            title="Model Analysis Assistant"
            description="Ask questions about model performance, get explanations, or request recommendations"
            placeholder="Ask about model performance, metrics interpretation, or recommendations..."
            onSubmit={handleSendMessage}
            isLoading={isLoading}
            messages={messages}
          />
        </div>
        
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Suggested Questions</CardTitle>
              <CardDescription>Try asking the AI assistant these questions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <SuggestedQuestion 
                question="Which is the best performing model?"
                onClick={() => handleSendMessage("Which is the best performing model?")}
              />
              <SuggestedQuestion 
                question="How much improvement does the best model show?"
                onClick={() => handleSendMessage("How much improvement does the best model show?")}
              />
              <SuggestedQuestion 
                question="Why did the LightGBM model fail?"
                onClick={() => handleSendMessage("Why did the LightGBM model fail?")}
              />
              <SuggestedQuestion 
                question="What metrics should I focus on for my use case?"
                onClick={() => handleSendMessage("What metrics should I focus on for my use case?")}
              />
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>AI Assistant Capabilities</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="list-disc pl-5 space-y-2 text-sm">
                <li>Analyze and explain model performance metrics</li>
                <li>Compare models and highlight key differences</li>
                <li>Identify potential issues and suggest improvements</li>
                <li>Explain complex ML concepts in simple terms</li>
                <li>Generate explanations of feature importance</li>
                <li>Recommend model selection based on use case</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function SuggestedQuestion({ question, onClick }: { question: string; onClick: () => void }) {
  return (
    <Button
      variant="outline"
      className="w-full justify-start text-left h-auto py-2 px-3"
      onClick={onClick}
    >
      {question}
    </Button>
  );
} 