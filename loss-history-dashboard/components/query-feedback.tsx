"use client"

import { useState, useEffect } from "react"
import { AlertCircle, CheckCircle, Info, RefreshCw } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AnimatedElement } from "@/components/animated-element"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { generateQueryExplanation, processQueryWithNLP } from "@/lib/enhanced-nlp-service"

interface QueryFeedbackProps {
  query: string
  parsedFilters: Record<string, any>
  className?: string
  onDismiss?: () => void
  onRephrase?: (suggestion: string) => void
}

export function QueryFeedback({ query, parsedFilters, className, onDismiss, onRephrase }: QueryFeedbackProps) {
  const [visible, setVisible] = useState(false)
  const [feedbackType, setFeedbackType] = useState<"success" | "warning" | "info">("info")
  const [message, setMessage] = useState("")
  const [suggestions, setSuggestions] = useState<string[]>([])

  useEffect(() => {
    if (!query) {
      setVisible(false)
      return
    }

    // Process the query with our enhanced NLP
    const nlpResult = processQueryWithNLP(query)
    const explanation = generateQueryExplanation(nlpResult)

    // Determine feedback type based on confidence and filters
    const filterCount = Object.keys(parsedFilters).length - (parsedFilters._originalQuery ? 1 : 0)
    const confidence = nlpResult.confidence || 0

    if (filterCount === 0 || confidence < 0.4) {
      setFeedbackType("warning")
      setMessage("I couldn't understand your query well. Try using different wording or check out the examples.")

      // Generate rephrasing suggestions
      setSuggestions([`Show data from last week`, `Find ${query} in recent runs`, `Search for ${query} in models`])
    } else if (confidence < 0.7 || (nlpResult.missingInformation && nlpResult.missingInformation.length > 0)) {
      setFeedbackType("info")
      setMessage(explanation)

      // Generate suggestions based on missing information
      if (nlpResult.missingInformation) {
        const rephraseOptions: string[] = []
        nlpResult.missingInformation.forEach((missing: any) => {
          if (missing.type === "date") {
            rephraseOptions.push(`${query} from last week`)
            rephraseOptions.push(`${query} in the last 30 days`)
          } else if (missing.type === "metric") {
            rephraseOptions.push(`${query} with accuracy above 0.9`)
            rephraseOptions.push(`${query} with high precision`)
          }
        })
        setSuggestions(rephraseOptions)
      }
    } else {
      setFeedbackType("success")
      setMessage(explanation)
      setSuggestions([])
    }

    setVisible(true)

    // Auto-hide after 10 seconds
    const timer = setTimeout(() => {
      setVisible(false)
    }, 10000)

    return () => clearTimeout(timer)
  }, [query, parsedFilters])

  if (!visible) return null

  return (
    <AnimatedElement
      animation={{
        initial: { opacity: 0, y: -10 },
        animate: { opacity: 1, y: 0 },
        exit: { opacity: 0, y: -10 },
        transition: { duration: 0.3 },
      }}
      className={cn("mb-4", className)}
    >
      <Alert
        variant={feedbackType === "success" ? "default" : feedbackType === "warning" ? "destructive" : "default"}
        className={cn(
          feedbackType === "info" &&
            "border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-300",
          "relative",
        )}
      >
        {feedbackType === "success" && <CheckCircle className="h-4 w-4" />}
        {feedbackType === "warning" && <AlertCircle className="h-4 w-4" />}
        {feedbackType === "info" && <Info className="h-4 w-4" />}

        <AlertTitle>
          {feedbackType === "success" && "Query Understood"}
          {feedbackType === "warning" && "Query Not Understood"}
          {feedbackType === "info" && "Query Partially Understood"}
        </AlertTitle>

        <AlertDescription className="space-y-2">
          <p>{message}</p>

          {suggestions.length > 0 && (
            <div className="pt-2">
              <p className="text-sm font-medium mb-1">Did you mean to ask:</p>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-1 text-xs"
                    onClick={() => onRephrase && onRephrase(suggestion)}
                  >
                    <RefreshCw className="h-3 w-3" />
                    {suggestion}
                  </Button>
                ))}
              </div>
            </div>
          )}
        </AlertDescription>

        {onDismiss && (
          <button
            onClick={onDismiss}
            className="absolute right-2 top-2 rounded-full p-1 text-foreground/50 opacity-70 transition-opacity hover:opacity-100"
            aria-label="Dismiss"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        )}
      </Alert>
    </AnimatedElement>
  )
}
