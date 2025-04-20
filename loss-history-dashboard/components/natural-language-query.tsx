"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Search, Sparkles, X, MessageSquareText, ThumbsUp, ThumbsDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  processQueryWithNLP,
  convertNLPResultToFilters,
  generateIntelligentSuggestions,
  generateQueryExplanation,
  expandKeywordQuery,
} from "@/lib/enhanced-nlp-service"
import { cn } from "@/lib/utils"
import { useAnimation } from "@/hooks/use-animation"
import { AnimatedElement } from "@/components/animated-element"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface NaturalLanguageQueryProps {
  onFilterChange: (filters: Record<string, any>) => void
  className?: string
}

export function NaturalLanguageQuery({ onFilterChange, className }: NaturalLanguageQueryProps) {
  const [query, setQuery] = useState("")
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [showExamples, setShowExamples] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [lastProcessedQuery, setLastProcessedQuery] = useState("")
  const [recentQueries, setRecentQueries] = useState<string[]>([])
  const [queryExplanation, setQueryExplanation] = useState<string>("")
  const [showExplanation, setShowExplanation] = useState(false)
  const [nlpResult, setNlpResult] = useState<any>(null)
  const [currentContext, setCurrentContext] = useState<string>("general")
  const [feedbackGiven, setFeedbackGiven] = useState(false)

  const inputRef = useRef<HTMLInputElement>(null)
  const suggestionsRef = useRef<HTMLDivElement>(null)
  const explanationRef = useRef<HTMLDivElement>(null)
  const { animate } = useAnimation()

  // Load recent queries from localStorage on component mount
  useEffect(() => {
    const savedQueries = localStorage.getItem("recentQueries")
    if (savedQueries) {
      try {
        setRecentQueries(JSON.parse(savedQueries))
      } catch (e) {
        console.error("Error loading recent queries:", e)
      }
    }

    // Detect context based on URL path
    const path = window.location.pathname
    if (path.includes("model")) {
      setCurrentContext("model-performance")
    } else if (path.includes("data")) {
      setCurrentContext("data-quality")
    } else {
      setCurrentContext("general")
    }
  }, [])

  // Save recent queries to localStorage when they change
  useEffect(() => {
    if (recentQueries.length > 0) {
      localStorage.setItem("recentQueries", JSON.stringify(recentQueries))
    }
  }, [recentQueries])

  useEffect(() => {
    // Generate suggestions based on current query
    if (query.trim().length > 1) {
      const newSuggestions = generateIntelligentSuggestions(query, recentQueries, {}, currentContext)
      setSuggestions(newSuggestions)
    } else {
      setSuggestions([])
    }
  }, [query, recentQueries, currentContext])

  useEffect(() => {
    // Close suggestions when clicking outside
    function handleClickOutside(event: MouseEvent) {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node) &&
        explanationRef.current &&
        !explanationRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false)
        setShowExamples(false)
        setShowExplanation(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [])

  const processQuery = (queryText: string) => {
    if (!queryText.trim() || queryText === lastProcessedQuery) return

    setIsProcessing(true)
    setShowSuggestions(false)
    setShowExamples(false)
    setFeedbackGiven(false)

    // Expand single keyword queries to natural language
    const expandedQuery = queryText.trim().split(/\s+/).length <= 1 ? expandKeywordQuery(queryText) : queryText

    // Process query with enhanced NLP
    setTimeout(() => {
      const result = processQueryWithNLP(expandedQuery)
      setNlpResult(result)

      const filters = convertNLPResultToFilters(result)
      const explanation = generateQueryExplanation(result)

      setQueryExplanation(explanation)
      onFilterChange(filters)
      setLastProcessedQuery(queryText)

      // Add to recent queries if not already there
      if (!recentQueries.includes(queryText)) {
        setRecentQueries((prevQueries) => [queryText, ...prevQueries].slice(0, 10))
      }

      setIsProcessing(false)
      setShowExplanation(true)

      // Animate the search icon to indicate successful processing
      animate("#search-icon", {
        scale: [1, 1.2, 1],
        transition: { duration: 0.3 },
      })
    }, 300)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      processQuery(query)
    } else if (e.key === "Escape") {
      setShowSuggestions(false)
      setShowExamples(false)
      setShowExplanation(false)
    } else if (e.key === "ArrowDown" && (showSuggestions || showExamples)) {
      e.preventDefault()
      const suggestionElements = document.querySelectorAll(".suggestion-item")
      if (suggestionElements.length > 0) {
        ;(suggestionElements[0] as HTMLElement).focus()
      }
    }
  }

  const handleSuggestionKeyDown = (e: React.KeyboardEvent, index: number, items: string[]) => {
    if (e.key === "Enter") {
      selectSuggestion(items[index])
    } else if (e.key === "ArrowDown") {
      e.preventDefault()
      const suggestionElements = document.querySelectorAll(".suggestion-item")
      if (index < suggestionElements.length - 1) {
        ;(suggestionElements[index + 1] as HTMLElement).focus()
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      if (index === 0) {
        inputRef.current?.focus()
      } else {
        const suggestionElements = document.querySelectorAll(".suggestion-item")
        ;(suggestionElements[index - 1] as HTMLElement).focus()
      }
    }
  }

  const selectSuggestion = (suggestion: string) => {
    setQuery(suggestion)
    setShowSuggestions(false)
    setShowExamples(false)
    processQuery(suggestion)
    inputRef.current?.focus()
  }

  const clearQuery = () => {
    setQuery("")
    setLastProcessedQuery("")
    setShowExplanation(false)
    onFilterChange({})
    inputRef.current?.focus()
  }

  const toggleExamples = () => {
    setShowExamples(!showExamples)
    setShowSuggestions(false)
  }

  const toggleExplanation = () => {
    setShowExplanation(!showExplanation)
  }

  const giveFeedback = (isPositive: boolean) => {
    if (nlpResult && !feedbackGiven) {
      if (!isPositive) {
        // For negative feedback, we could show a form to collect more information
        // But for now, just log it
        console.log("Negative feedback for query:", query)
      } else {
        // Positive feedback can help reinforce good interpretations
        console.log("Positive feedback for query:", query)
      }

      setFeedbackGiven(true)

      // Animate feedback button
      animate(isPositive ? "#thumbs-up" : "#thumbs-down", {
        scale: [1, 1.5, 1],
        transition: { duration: 0.4 },
      })
    }
  }

  return (
    <div className={cn("relative", className)}>
      <div className="relative flex items-center">
        <div className="absolute left-3 top-1/2 -translate-y-1/2">
          <AnimatedElement id="search-icon">
            <Search className="h-4 w-4 text-muted-foreground" />
          </AnimatedElement>
        </div>

        <Input
          ref={inputRef}
          type="text"
          placeholder="Ask anything about your dashboard data (e.g., 'Show failed runs from last week')"
          className="pl-9 pr-24 h-10"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => {
            if (query.trim().length > 0) {
              setShowSuggestions(true)
            } else {
              setShowExamples(true)
            }
          }}
          onKeyDown={handleKeyDown}
        />

        <div className="absolute right-2 flex items-center space-x-1">
          {query.length > 0 && (
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={clearQuery} title="Clear query">
              <X className="h-3.5 w-3.5" />
            </Button>
          )}

          {lastProcessedQuery && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={toggleExplanation}
                    title="Show how I understood your query"
                  >
                    <MessageSquareText className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Show how I understood your query</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}

          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={toggleExamples} title="Show examples">
            <Sparkles className="h-3.5 w-3.5" />
          </Button>

          <Button
            variant="default"
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => processQuery(query)}
            disabled={isProcessing || !query.trim() || query === lastProcessedQuery}
          >
            {isProcessing ? "Processing..." : "Search"}
          </Button>
        </div>
      </div>

      {/* Suggestions dropdown */}
      {showSuggestions && suggestions.length > 0 && (
        <div ref={suggestionsRef} className="absolute z-50 mt-1 w-full rounded-md border bg-popover p-2 shadow-md">
          <div className="text-xs font-medium text-muted-foreground mb-1.5">Suggestions</div>
          <div className="space-y-0.5">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                className="suggestion-item w-full rounded-sm px-2 py-1.5 text-sm text-left hover:bg-accent focus:bg-accent focus:outline-none"
                onClick={() => selectSuggestion(suggestion)}
                onKeyDown={(e) => handleSuggestionKeyDown(e, index, suggestions)}
                tabIndex={0}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Examples dropdown */}
      {showExamples && (
        <div ref={suggestionsRef} className="absolute z-50 mt-1 w-full rounded-md border bg-popover p-2 shadow-md">
          <div className="text-xs font-medium text-muted-foreground mb-1.5">Try these examples</div>
          <div className="space-y-0.5">
            {generateIntelligentSuggestions("", recentQueries, {}, currentContext).map((example, index) => (
              <button
                key={index}
                className="suggestion-item w-full rounded-sm px-2 py-1.5 text-sm text-left hover:bg-accent focus:bg-accent focus:outline-none"
                onClick={() => selectSuggestion(example)}
                onKeyDown={(e) => handleSuggestionKeyDown(e, index, suggestions)}
                tabIndex={0}
              >
                {example}
              </button>
            ))}
          </div>

          {recentQueries.length > 0 && (
            <>
              <div className="text-xs font-medium text-muted-foreground mb-1.5 mt-3">Recent queries</div>
              <div className="space-y-0.5">
                {recentQueries.slice(0, 3).map((recentQuery, index) => (
                  <button
                    key={`recent-${index}`}
                    className="suggestion-item w-full rounded-sm px-2 py-1.5 text-sm text-left hover:bg-accent focus:bg-accent focus:outline-none"
                    onClick={() => selectSuggestion(recentQuery)}
                    tabIndex={0}
                  >
                    {recentQuery}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* Explanation popover */}
      {showExplanation && queryExplanation && (
        <div ref={explanationRef} className="absolute z-50 mt-1 w-full rounded-md border bg-popover p-3 shadow-md">
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-sm font-medium">How I understood your query</h4>
            <div className="flex items-center space-x-1">
              {!feedbackGiven && (
                <>
                  <AnimatedElement id="thumbs-up">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => giveFeedback(true)}
                      className="h-6 w-6"
                      title="This interpretation was correct"
                    >
                      <ThumbsUp className="h-3.5 w-3.5 text-muted-foreground" />
                    </Button>
                  </AnimatedElement>

                  <AnimatedElement id="thumbs-down">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => giveFeedback(false)}
                      className="h-6 w-6"
                      title="This interpretation was incorrect"
                    >
                      <ThumbsDown className="h-3.5 w-3.5 text-muted-foreground" />
                    </Button>
                  </AnimatedElement>
                </>
              )}

              <Button variant="ghost" size="icon" onClick={() => setShowExplanation(false)} className="h-6 w-6">
                <X className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>

          <div className="text-sm whitespace-pre-line text-muted-foreground">{queryExplanation}</div>
        </div>
      )}
    </div>
  )
}
