type FilterType = "date" | "status" | "model" | "severity" | "search" | "metric"

interface ParsedFilter {
  type: FilterType
  value: any
  confidence: number // 0-1 indicating how confident we are in this interpretation
}

interface ParsedQuery {
  filters: ParsedFilter[]
  originalQuery: string
  normalizedQuery: string
}

// Date-related keywords and patterns
const dateKeywords = {
  today: /today|24 hours/i,
  yesterday: /yesterday/i,
  thisWeek: /this week|current week/i,
  lastWeek: /last week|previous week/i,
  thisMonth: /this month|current month/i,
  lastMonth: /last month|previous month/i,
  thisYear: /this year|current year/i,
  lastYear: /last year|previous year/i,
  days: /(\d+)\s*(day|days)/i,
  weeks: /(\d+)\s*(week|weeks)/i,
  months: /(\d+)\s*(month|months)/i,
  years: /(\d+)\s*(year|years)/i,
  dateRange: /from\s(.+?)\sto\s(.+)/i,
  since: /since\s(.+)/i,
  before: /before\s(.+)/i,
  after: /after\s(.+)/i,
}

// Status-related keywords
const statusKeywords = {
  success: /success|successful|completed|passed/i,
  failed: /fail|failed|error|errors/i,
  warning: /warning|warnings|warn/i,
  running: /running|in progress|ongoing/i,
  pending: /pending|waiting|queued/i,
}

// Severity-related keywords
const severityKeywords = {
  critical: /critical|severe|high priority|urgent/i,
  warning: /warning|medium priority|attention/i,
  info: /info|information|low priority/i,
}

// Model-related keywords
const modelKeywords = {
  gradientBoosting: /gradient\s*boost|gbm|gbdt/i,
  randomForest: /random\s*forest|rf/i,
  xgboost: /xgboost|xgb/i,
  neuralNetwork: /neural\s*network|nn|deep\s*learning|dl/i,
  lightGBM: /lightgbm|lgbm/i,
}

// Metric-related keywords
const metricKeywords = {
  accuracy: /accuracy/i,
  precision: /precision/i,
  recall: /recall/i,
  f1: /f1|f1\s*score/i,
  auc: /auc|area\s*under\s*curve|roc/i,
  rmse: /rmse|root\s*mean\s*squared\s*error/i,
  mae: /mae|mean\s*absolute\s*error/i,
}

// Common query patterns
const queryPatterns = [
  // Date patterns
  { pattern: /show me (data|results) from (today|yesterday|this week|last week|this month|last month)/i, type: "date" },
  { pattern: /show (data|results) from the last (\d+) (days|weeks|months|years)/i, type: "date" },
  { pattern: /(data|results) (from|since|after) (.+)/i, type: "date" },
  { pattern: /(data|results) before (.+)/i, type: "date" },

  // Status patterns
  {
    pattern: /show (me )?(only )?(success|successful|failed|error|warning|running) (runs|jobs|pipelines)/i,
    type: "status",
  },
  { pattern: /filter by (success|successful|failed|error|warning|running) status/i, type: "status" },

  // Model patterns
  {
    pattern: /show (me )?(only )?(gradient boost|random forest|xgboost|neural network|lightgbm) models/i,
    type: "model",
  },
  { pattern: /filter by (gradient boost|random forest|xgboost|neural network|lightgbm) model/i, type: "model" },

  // Severity patterns
  { pattern: /show (me )?(only )?(critical|warning|info) (alerts|issues|problems)/i, type: "severity" },
  { pattern: /(critical|warning|info) (alerts|issues|problems) only/i, type: "severity" },

  // Metric patterns
  {
    pattern:
      /(models|runs) with (accuracy|precision|recall|f1|auc|rmse|mae) (above|below|greater than|less than) (\d+\.?\d*)/i,
    type: "metric",
  },

  // Search patterns
  { pattern: /search for (.+)/i, type: "search" },
  { pattern: /find (.+)/i, type: "search" },
  { pattern: /containing (.+)/i, type: "search" },
]

/**
 * Parse a natural language query into structured filter parameters
 */
export function parseNaturalLanguageQuery(query: string): ParsedQuery {
  const normalizedQuery = query.toLowerCase().trim()
  const filters: ParsedFilter[] = []

  // Try to match date-related queries
  if (dateKeywords.today.test(normalizedQuery)) {
    const today = new Date()
    const tomorrow = new Date()
    tomorrow.setDate(today.getDate() + 1)
    filters.push({
      type: "date",
      value: { from: today, to: tomorrow },
      confidence: 0.9,
    })
  } else if (dateKeywords.yesterday.test(normalizedQuery)) {
    const yesterday = new Date()
    yesterday.setDate(yesterday.getDate() - 1)
    const today = new Date()
    filters.push({
      type: "date",
      value: { from: yesterday, to: today },
      confidence: 0.9,
    })
  } else if (dateKeywords.thisWeek.test(normalizedQuery)) {
    const today = new Date()
    const startOfWeek = new Date()
    startOfWeek.setDate(today.getDate() - today.getDay())
    filters.push({
      type: "date",
      value: { from: startOfWeek, to: today },
      confidence: 0.9,
    })
  } else if (dateKeywords.lastWeek.test(normalizedQuery)) {
    const today = new Date()
    const endOfLastWeek = new Date()
    endOfLastWeek.setDate(today.getDate() - today.getDay())
    const startOfLastWeek = new Date(endOfLastWeek)
    startOfLastWeek.setDate(endOfLastWeek.getDate() - 7)
    filters.push({
      type: "date",
      value: { from: startOfLastWeek, to: endOfLastWeek },
      confidence: 0.9,
    })
  } else if (dateKeywords.thisMonth.test(normalizedQuery)) {
    const today = new Date()
    const startOfMonth = new Date(today.getFullYear(), today.getMonth(), 1)
    filters.push({
      type: "date",
      value: { from: startOfMonth, to: today },
      confidence: 0.9,
    })
  } else if (dateKeywords.lastMonth.test(normalizedQuery)) {
    const today = new Date()
    const endOfLastMonth = new Date(today.getFullYear(), today.getMonth(), 0)
    const startOfLastMonth = new Date(today.getFullYear(), today.getMonth() - 1, 1)
    filters.push({
      type: "date",
      value: { from: startOfLastMonth, to: endOfLastMonth },
      confidence: 0.9,
    })
  } else {
    // Check for "last X days/weeks/months/years" pattern
    const daysMatch = normalizedQuery.match(dateKeywords.days)
    if (daysMatch && daysMatch[1]) {
      const days = Number.parseInt(daysMatch[1], 10)
      const today = new Date()
      const pastDate = new Date()
      pastDate.setDate(today.getDate() - days)
      filters.push({
        type: "date",
        value: { from: pastDate, to: today },
        confidence: 0.85,
      })
    }

    const weeksMatch = normalizedQuery.match(dateKeywords.weeks)
    if (weeksMatch && weeksMatch[1]) {
      const weeks = Number.parseInt(weeksMatch[1], 10)
      const today = new Date()
      const pastDate = new Date()
      pastDate.setDate(today.getDate() - weeks * 7)
      filters.push({
        type: "date",
        value: { from: pastDate, to: today },
        confidence: 0.85,
      })
    }

    const monthsMatch = normalizedQuery.match(dateKeywords.months)
    if (monthsMatch && monthsMatch[1]) {
      const months = Number.parseInt(monthsMatch[1], 10)
      const today = new Date()
      const pastDate = new Date()
      pastDate.setMonth(today.getMonth() - months)
      filters.push({
        type: "date",
        value: { from: pastDate, to: today },
        confidence: 0.85,
      })
    }
  }

  // Try to match status-related queries
  Object.entries(statusKeywords).forEach(([status, regex]) => {
    if (regex.test(normalizedQuery)) {
      filters.push({
        type: "status",
        value: [status],
        confidence: 0.8,
      })
    }
  })

  // Try to match severity-related queries
  Object.entries(severityKeywords).forEach(([severity, regex]) => {
    if (regex.test(normalizedQuery)) {
      filters.push({
        type: "severity",
        value: severity,
        confidence: 0.8,
      })
    }
  })

  // Try to match model-related queries
  const modelMatches: string[] = []
  Object.entries(modelKeywords).forEach(([model, regex]) => {
    if (regex.test(normalizedQuery)) {
      modelMatches.push(
        model === "gradientBoosting"
          ? "model1"
          : model === "randomForest"
            ? "model2"
            : model === "xgboost"
              ? "model3"
              : model === "neuralNetwork"
                ? "model4"
                : "model5",
      )
    }
  })

  if (modelMatches.length > 0) {
    filters.push({
      type: "model",
      value: modelMatches,
      confidence: 0.8,
    })
  }

  // Check for search terms
  const searchMatch =
    normalizedQuery.match(/search for (.+)/i) ||
    normalizedQuery.match(/find (.+)/i) ||
    normalizedQuery.match(/containing (.+)/i)

  if (searchMatch && searchMatch[1]) {
    filters.push({
      type: "search",
      value: searchMatch[1],
      confidence: 0.7,
    })
  }

  // If no specific patterns matched but there are words, use them as search terms
  if (filters.length === 0 && normalizedQuery.length > 0) {
    // Exclude common words that are likely not search terms
    const commonWords = [
      "show",
      "me",
      "get",
      "find",
      "display",
      "list",
      "all",
      "the",
      "with",
      "and",
      "or",
      "in",
      "on",
      "at",
    ]
    const searchTerms = normalizedQuery
      .split(" ")
      .filter((word) => !commonWords.includes(word) && word.length > 2)
      .join(" ")

    if (searchTerms.length > 0) {
      filters.push({
        type: "search",
        value: searchTerms,
        confidence: 0.5,
      })
    }
  }

  return {
    filters,
    originalQuery: query,
    normalizedQuery,
  }
}

/**
 * Convert parsed query to filter parameters for the dashboard
 */
export function convertParsedQueryToFilters(parsedQuery: ParsedQuery): Record<string, any> {
  const result: Record<string, any> = {}

  parsedQuery.filters.forEach((filter) => {
    // Only use filters with confidence above a threshold
    if (filter.confidence >= 0.5) {
      result[filter.type] = filter.value
    }
  })

  return result
}

/**
 * Generate suggestions based on partial query
 */
export function generateQuerySuggestions(partialQuery: string): string[] {
  const suggestions: string[] = []
  const normalizedQuery = partialQuery.toLowerCase().trim()

  // Date-related suggestions
  if (normalizedQuery.includes("date") || normalizedQuery.includes("time") || normalizedQuery.includes("day")) {
    suggestions.push("Show me data from today")
    suggestions.push("Show me data from yesterday")
    suggestions.push("Show me data from this week")
    suggestions.push("Show me data from the last 7 days")
    suggestions.push("Show me data from the last 30 days")
  }

  // Status-related suggestions
  if (normalizedQuery.includes("status") || normalizedQuery.includes("state")) {
    suggestions.push("Show only successful runs")
    suggestions.push("Show only failed jobs")
    suggestions.push("Show only running pipelines")
    suggestions.push("Show only warning alerts")
  }

  // Model-related suggestions
  if (normalizedQuery.includes("model")) {
    suggestions.push("Show only Gradient Boosting models")
    suggestions.push("Show only Random Forest models")
    suggestions.push("Show only XGBoost models")
    suggestions.push("Show only Neural Network models")
  }

  // Severity-related suggestions
  if (normalizedQuery.includes("severity") || normalizedQuery.includes("priority")) {
    suggestions.push("Show only critical alerts")
    suggestions.push("Show only warning issues")
    suggestions.push("Show only info notifications")
  }

  // If no specific category is detected, provide general suggestions
  if (suggestions.length === 0) {
    suggestions.push("Show me data from the last 7 days")
    suggestions.push("Show only successful runs")
    suggestions.push("Show only failed jobs")
    suggestions.push("Show only critical alerts")
    suggestions.push("Search for specific model name")
  }

  return suggestions
}

/**
 * Get example queries to help users
 */
export function getExampleQueries(): string[] {
  return [
    "Show me data from the last 7 days",
    "Show only successful runs",
    "Show only failed jobs from yesterday",
    "Show critical alerts from this week",
    "Show Gradient Boosting models with accuracy above 0.9",
    'Find runs containing "experiment-1"',
    "Show Neural Network models from last month",
    "Show only warning issues from today",
  ]
}
