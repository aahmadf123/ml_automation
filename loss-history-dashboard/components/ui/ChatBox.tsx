import React, { useState } from 'react';
import { Button } from './button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './card';
import { Textarea } from './textarea';
import { ChevronRight, Bot, Send, Loader2 } from 'lucide-react';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
}

export interface ChatBoxProps {
  title?: string;
  description?: string;
  placeholder?: string;
  onSubmit: (message: string) => void;
  isLoading?: boolean;
  messages?: ChatMessage[];
  className?: string;
}

export function ChatBox({
  title = "AI Assistant",
  description = "Ask questions about model performance or request analyses",
  placeholder = "Enter your question about the models...",
  onSubmit,
  isLoading = false,
  messages = [],
  className = "",
}: ChatBoxProps) {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onSubmit(inputValue.trim());
      setInputValue('');
    }
  };

  return (
    <Card className={`border shadow-sm w-full ${className}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl flex items-center">
              <Bot className="mr-2 h-5 w-5" /> {title}
            </CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {messages.length > 0 && (
          <div className="mb-4 max-h-[300px] overflow-y-auto border rounded-md p-3 bg-gray-50">
            {messages.map((message, index) => (
              <div 
                key={index} 
                className={`mb-3 ${
                  message.role === 'user' ? 'ml-auto' : ''
                }`}
              >
                <div 
                  className={`rounded-md p-3 max-w-[80%] inline-block ${
                    message.role === 'user' 
                      ? 'bg-blue-100 text-blue-800' 
                      : 'bg-white border text-gray-800'
                  }`}
                >
                  {message.content}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {message.role === 'user' ? 'You' : 'AI Assistant'} 
                  {message.timestamp && ` • ${message.timestamp.toLocaleTimeString()}`}
                </div>
              </div>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-2">
          <Textarea 
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={placeholder}
            className="min-h-[100px] w-full resize-none"
            disabled={isLoading}
          />
          <div className="flex justify-end">
            <Button 
              type="submit" 
              disabled={!inputValue.trim() || isLoading}
              className="flex items-center gap-1"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" />
                  Send
                </>
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
} 