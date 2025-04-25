"use client";

import type { ParsedFilter } from "./nl-query-parser"

// Entity type definitions for named entity recognition
export type EntityType =
  | "date"
  | "model"
  | "status"
  | "metric"
  | "severity"
  | "threshold"
  | "field"
  | "value"
  | "comparison"
  | "location"

export interface DomainKnowledge {
tent {
  type: IntentType
  confidence: number
  entities: Entity[]
}

// Semantic vocabulary for understanding related terms
export interface SemanticTerm {
mber
}

// Intent definitions for understanding query purpose
export type IntentType =
  | "filter"
  | "sort"
  | "show"
  | "compare"
  | "trend"
  | "summarize"
  | "count"
  | "group"
  | "find_anomalies"
  | "predict"

interface Intent {
  type: IntentType
  confidence: number
  entities: Entity[]
}

// Semantic vocabulary for understanding related terms
interface SemanticTerm {
  term: string
  synonyms: string[]
  related: string[]
  category: string
}

// Result of NLP processing
export interface NLPResult {
  query: string
  normalizedQuery: string
  intents: Intent[]
  entities: Entity[]
  filters: ParsedFilter[]
  sortOptions?: {
    field: string
    direction: "asc" | "desc"
  }
  confidence: number
  missingInformation?: {
    type: string
    message: string
  }[]
  correctedQuery?: string
}

// Add these new types at the top of the file, after the existing type definitions

// Context awareness for better query understanding
export interface QueryContext {
  currentPage: string
  recentSearches: string[]
  userPreferences?: Record<string, any>
  domainKnowledge: DomainKnowledge
}

interface DomainKnowledge {
  entities: Record<string, string[]>
  relationships: Record<string, string[]>
  synonyms: Record<string, string[]>
  domainSpecificTerms: Record<string, any>
  locations: Record<string, any>
}

// The semantic vocabulary helps understand related terms
const semanticVocabulary: Record<string, SemanticTerm> = {
  // Model types
  "gradient boosting": {
    term: "gradient boosting",
    synonyms: ["gbm", "gbdt", "gb", "gradient boost", "boosted trees"],
    related: ["xgboost", "lightgbm", "catboost"],
    category: "model",
  },
  "random forest": {
    term: "random forest",
    synonyms: ["rf", "random forests", "forest"],
    related: ["decision trees", "bagging"],
    category: "model",
  },
  xgboost: {
    term: "xgboost",
    synonyms: ["xgb", "extreme gradient boosting"],
    related: ["gradient boosting", "lightgbm"],
    category: "model",
  },
  "neural network": {
    term: "neural network",
    synonyms: ["nn", "deep learning", "dl", "neural net", "artificial neural network", "ann"],
    related: ["deep neural network", "cnn", "rnn", "transformer"],
    category: "model",
  },
  lightgbm: {
    term: "lightgbm",
    synonyms: ["lgbm", "light gbm", "light gradient boosting machine"],
    related: ["xgboost", "gradient boosting"],
    category: "model",
  },

  // Status terms
  success: {
    term: "success",
    synonyms: ["successful", "succeeded", "completed", "passed", "ok", "good", "green"],
    related: ["finished", "done", "positive"],
    category: "status",
  },
  failure: {
    term: "failure",
    synonyms: ["fail", "failed", "error", "errors", "crashed", "broken", "red"],
    related: ["exception", "problem", "issue", "bug"],
    category: "status",
  },
  warning: {
    term: "warning",
    synonyms: ["warnings", "warn", "caution", "yellow", "orange"],
    related: ["alert", "attention", "notice"],
    category: "status",
  },
  running: {
    term: "running",
    synonyms: ["in progress", "ongoing", "executing", "processing", "active", "blue"],
    related: ["started", "busy", "working"],
    category: "status",
  },
  pending: {
    term: "pending",
    synonyms: ["waiting", "queued", "scheduled", "in queue", "gray"],
    related: ["delayed", "on hold", "standby"],
    category: "status",
  },

  // Severity terms
  critical: {
    term: "critical",
    synonyms: ["severe", "high priority", "urgent", "major", "important", "p0", "p1"],
    related: ["emergency", "crucial", "vital"],
    category: "severity",
  },
  warning: {
    term: "warning",
    synonyms: ["medium priority", "attention", "moderate", "p2"],
    related: ["caution", "careful", "heads up"],
    category: "severity",
  },
  info: {
    term: "info",
    synonyms: ["information", "low priority", "minor", "p3", "trivial"],
    related: ["notification", "update", "fyi"],
    category: "severity",
  },

  // Date terms
  today: {
    term: "today",
    synonyms: ["current day", "this day", "24 hours", "past 24h", "past 24 hours"],
    related: ["now", "current", "recent"],
    category: "date",
  },
  yesterday: {
    term: "yesterday",
    synonyms: ["previous day", "day before", "last day"],
    related: ["recent", "past", "day ago"],
    category: "date",
  },
  week: {
    term: "week",
    synonyms: ["7 days", "weekly", "this week", "current week"],
    related: ["days", "weekly", "workweek"],
    category: "date",
  },
  month: {
    term: "month",
    synonyms: ["30 days", "monthly", "this month", "current month"],
    related: ["days", "monthly"],
    category: "date",
  },

  // Comparison terms
  above: {
    term: "above",
    synonyms: ["greater than", ">", "more than", "higher than", "exceeding", "over"],
    related: ["greater", "higher", "larger"],
    category: "comparison",
  },
  below: {
    term: "below",
    synonyms: ["less than", "<", "lower than", "under", "beneath", "smaller than"],
    related: ["less", "lower", "smaller"],
    category: "comparison",
  },
  equal: {
    term: "equal",
    synonyms: ["equals", "=", "same as", "exactly", "precisely"],
    related: ["identical", "matching", "equivalent"],
    category: "comparison",
  },

  // Metric terms
  accuracy: {
    term: "accuracy",
    synonyms: ["acc", "correct rate", "correctness"],
    related: ["precision", "recall", "f1"],
    category: "metric",
  },
  precision: {
    term: "precision",
    synonyms: ["prec", "positive predictive value"],
    related: ["accuracy", "recall", "f1"],
    category: "metric",
  },
  recall: {
    term: "recall",
    synonyms: ["sensitivity", "true positive rate", "hit rate"],
    related: ["accuracy", "precision", "f1"],
    category: "metric",
  },
  f1: {
    term: "f1",
    synonyms: ["f1 score", "f-score", "f-measure"],
    related: ["accuracy", "precision", "recall"],
    category: "metric",
  },
  auc: {
    term: "auc",
    synonyms: ["area under curve", "roc auc", "auc-roc", "area under roc"],
    related: ["roc", "precision-recall curve"],
    category: "metric",
  },
  rmse: {
    term: "rmse",
    synonyms: ["root mean squared error", "root mean square error"],
    related: ["mse", "mae", "error metrics"],
    category: "metric",
  },
  ohio: {
    term: "ohio",
    synonyms: ["oh", "columbus"],
    related: ["insurance", "rates", "claims"],
    category: "location",
  },
}

/**
 * Finds semantically related terms in the query
 */
function findSemanticTerms(query: string): Map<string, Entity[]> {
  const normalizedQuery = query.toLowerCase()
  const result = new Map<string, Entity[]>()

  // For each term in our vocabulary
  Object.entries(semanticVocabulary).forEach(([key, termData]) => {
    const allTerms = [termData.term, ...termData.synonyms]
    const foundEntities: Entity[] = []

    // Check if any of the terms or synonyms are in the query
    allTerms.forEach((term) => {
      const regex = new RegExp(`\\b${term}\\b`, "gi")
      let match
      while ((match = regex.exec(normalizedQuery)) !== null) {
        foundEntities.push({
          text: match[0],
          type: termData.category as EntityType,
          startPosition: match.index,
          endPosition: match.index + match[0].length,
          value: key, // Use the canonical term as the value
          confidence: term === termData.term ? 0.95 : 0.85, // Higher confidence for exact matches
        })
      }
    })

    // Add the found entities if any
    if (foundEntities.length > 0) {
      result.set(key, foundEntities)
    }
  })

  return result
}

/**
 * Identify named entities in the query
 */
function extractEntities(query: string): Entity[] {
  const entities: Entity[] = []
  const normalizedQuery = query.toLowerCase()

  // Extract semantic terms first
  const semanticTerms = findSemanticTerms(query)
  semanticTerms.forEach((termEntities) => {
    entities.push(...termEntities)
  })

  // Extract dates with regex patterns
  const datePatterns = [
    {
      regex: /\b(?:from|since|after)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b/i,
      type: "date" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        const dateStr = match[1]
        try {
          return { from: new Date(dateStr), to: null }
        } catch (e) {
          return { text: dateStr }
        }
      },
    },
    {
      regex: /\b(?:to|until|before)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b/i,
      type: "date" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        const dateStr = match[1]
        try {
          return { from: null, to: new Date(dateStr) }
        } catch (e) {
          return { text: dateStr }
        }
      },
    },
    {
      regex: /\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+to\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b/i,
      type: "date" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        try {
          return { from: new Date(match[1]), to: new Date(match[2]) }
        } catch (e) {
          return { text: `${match[1]} to ${match[2]}` }
        }
      },
    },
    {
      regex: /\blast\s+(\d+)с+(day|days|week|weeks|month|months|year|years)\b/i,
      type: "date" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        const num = Number.parseInt(match[1], 10)
        const unit = match[2].toLowerCase()
        const to = new Date()
        const from = new Date()

        if (unit === "day" || unit === "days") {
          from.setDate(from.getDate() - num)
        } else if (unit === "week" || unit === "weeks") {
          from.setDate(from.getDate() - num * 7)
        } else if (unit === "month" || unit === "months") {
          from.setMonth(from.getMonth() - num)
        } else if (unit === "year" || unit === "years") {
          from.setFullYear(from.getFullYear() - num)
        }

        return { from, to }
      },
    },
  ]

  // Apply date patterns
  datePatterns.forEach((pattern) => {
    const regex = pattern.regex
    let match
    while ((match = regex.exec(normalizedQuery)) !== null) {
      const value = pattern.valueExtractor(match)
      entities.push({
        text: match[0],
        type: pattern.type,
        startPosition: match.index,
        endPosition: match.index + match[0].length,
        value,
        confidence: 0.9,
      })
    }
  })

  // Extract comparison expressions with regex
  const comparisonPatterns = [
    {
      regex:
        /\b(accuracy|precision|recall|f1|auc|rmse|mae)\s+(above|greater than|more than|higher than|over|exceeding)\s+(\d+\.?\d*)\b/i,
      type: "comparison" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        return {
          metric: match[1].toLowerCase(),
          operator: ">",
          value: Number.parseFloat(match[3]),
        }
      },
    },
    {
      regex:
        /\b(accuracy|precision|recall|f1|auc|rmse|mae)\s+(below|less than|lower than|under|beneath|smaller than)\s+(\d+\.?\d*)\b/i,
      type: "comparison" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        return {
          metric: match[1].toLowerCase(),
          operator: "<",
          value: Number.parseFloat(match[3]),
        }
      },
    },
    {
      regex: /\b(accuracy|precision|recall|f1|auc|rmse|mae)\s+(equals|equal to|=|is)\s+(\d+\.?\d*)\b/i,
      type: "comparison" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        return {
          metric: match[1].toLowerCase(),
          operator: "=",
          value: Number.parseFloat(match[3]),
        }
      },
    },
  ]

  // Apply comparison patterns
  comparisonPatterns.forEach((pattern) => {
    const regex = pattern.regex
    let match
    while ((match = regex.exec(normalizedQuery)) !== null) {
      const value = pattern.valueExtractor(match)
      entities.push({
        text: match[0],
        type: pattern.type,
        startPosition: match.index,
        endPosition: match.index + match[0].length,
        value,
        confidence: 0.85,
      })
    }
  })

  // Extract location entities with regex
  const locationPatterns = [
    {
      regex: /\bohio\b/i,
      type: "location" as EntityType,
      valueExtractor: () => {
        return {
          state: "Ohio",
        }
      },
    },
  ]

  // Apply location patterns
  locationPatterns.forEach((pattern) => {
    const regex = pattern.regex
    let match
    while ((match = regex.exec(normalizedQuery)) !== null) {
      const value = pattern.valueExtractor(match)
      entities.push({
        text: match[0],
        type: pattern.type,
        startPosition: match.index,
        endPosition: match.index + match[0].length,
        value,
        confidence: 0.9,
      })
    }
  })

  // Sort entities by position in the text
  return explanation
},

/**
 * Convert a search query to a full natural language query
 * This helps transform simple keyword searches into proper natural language
 */
expandKeywordQuery(query: string): string {
  const intents: Intent[] = []

  // Intent patterns with their regexes and confidence scores
  const intentPatterns = [
    {
      type: "filter" as IntentType,
      patterns: [
        { regex: /\b(?:filter|show|display|find|get|give me|where)\b/i, confidence: 0.8 },
        { regex: /\bonly\b/i, confidence: 0.7 },
      ],
    },
    {
      type: "sort" as IntentType,
      patterns: [
        { regex: /\b(?:sort|order|arrange|rank)\b.+\b(?:by|on|using)\b/i, confidence: 0.9 },
        { regex: /\bin\s+(?:ascending|descending)\s+order\b/i, confidence: 0.85 },
        { regex: /\bhighest|lowest|best|worst\b/i, confidence: 0.7 },
      ],
    },
    {
      type: "compare" as IntentType,
      patterns: [
        { regex: /\b(?:compare|comparison|versus|vs|against)\b/i, confidence: 0.9 },
        { regex: /\b(?:difference|different|changes|changed)\b/i, confidence: 0.7 },
      ],
    },
    {
      type: "trend" as IntentType,
      patterns: [
        { regex: /\b(?:trend|trends|trending|over time|time series|evolution|progress)\b/i, confidence: 0.9 },
        { regex: /\b(?:increase|decrease|growth|decline|change)\b.+\btime\b/i, confidence: 0.8 },
      ],
    },
    {
      type: "summarize" as IntentType,
      patterns: [
        {
          regex: /\b(?:summarize|summary|overview|brief|snapshot|synopsis|summary of|overview of)\b/i,
          confidence: 0.9,
        },
        { regex: /\bkey\s+(?:metrics|statistics|numbers|figures|insights)\b/i, confidence: 0.85 },
      ],
    },
    {
      type: "count" as IntentType,
      patterns: [
        { regex: /\b(?:count|how many|number of|total|sum)\b/i, confidence: 0.85 },
        { regex: /\b(?:count|tally)\b.+\b(?:by|per|for each)\b/i, confidence: 0.8 },
      ],
    },
    {
      type: "group" as IntentType,
      patterns: [
        { regex: /\b(?:group|categorize|segment|bucket|cluster)\b.+\b(?:by|into|using)\b/i, confidence: 0.9 },
        { regex: /\b(?:grouped|categorized)\b.+\b(?:by|into)\b/i, confidence: 0.85 },
      ],
    },
    {
      type: "find_anomalies" as IntentType,
      patterns: [
        { regex: /\b(?:anomaly|anomalies|outlier|outliers|unusual|abnormal|irregular)\b/i, confidence: 0.9 },
        {
          regex: /\b(?:detect|identify|find|discover)\b.+\b(?:anomaly|anomalies|outlier|outliers)\b/i,
          confidence: 0.85,
        },
      ],
    },
    {
      type: "predict" as IntentType,
      patterns: [
        { regex: /\b(?:predict|forecast|projection|estimate|projection)\b/i, confidence: 0.9 },
        { regex: /\b(?:will|future|next|upcoming|coming)\b.+\b(?:predict|forecast|projection)\b/i, confidence: 0.8 },
      ],
    },
  ]

  // Test each intent pattern against the normalized query
  intentPatterns.forEach((intent) => {
    for (const pattern of intent.patterns) {
      if (pattern.regex.test(normalizedQuery)) {
        intents.push({
          type: intent.type,
          confidence: pattern.confidence,
          entities: entities.filter((e) => e.confidence > 0.7), // Only include high-confidence entities
        })
        break // Once we find a match for this intent type, we don't need to check other patterns
      }
    }
  })

  // If no specific intent is found, default to "filter" as the most basic intent
  if (intents.length === 0) {
    intents.push({
      type: "filter",
      confidence: 0.6, // Lower confidence for default intent
      entities,
    })
  }

  // Sort intents by confidence (highest first)
  return intents.sort((a, b) => b.confidence - a.confidence)
}

/**
 * Convert entities to filter parameters
 */
function entitiesToFilters(entities: Entity[]): ParsedFilter[] {
  const filters: ParsedFilter[] = []
  const usedEntities = new Set<Entity>()

  // Process date entities
  const dateEntities = entities.filter((e) => e.type === "date")
  if (dateEntities.length > 0) {
    // Combine date entities if possible
    let fromDate = null
    let toDate = null

    for (const entity of dateEntities) {
      if (entity.value.from) {
        fromDate = entity.value.from
      }
      if (entity.value.to) {
        toDate = entity.value.to
      }
      usedEntities.add(entity)
    }

    if (fromDate || toDate) {
      filters.push({
        type: "date",
        value: { from: fromDate, to: toDate },
        confidence: 0.9,
      })
    }
  }

  // Process status entities
  const statusEntities = entities.filter((e) => e.type === "status" && !usedEntities.has(e))
  if (statusEntities.length > 0) {
    const statuses = statusEntities.map((e) => e.value)
    filters.push({
      type: "status",
      value: statuses,
      confidence: statusEntities.reduce((acc, e) => acc + e.confidence, 0) / statusEntities.length,
    })
    statusEntities.forEach((e) => usedEntities.add(e))
  }

  // Process model entities
  const modelEntities = entities.filter((e) => e.type === "model" && !usedEntities.has(e))
  if (modelEntities.length > 0) {
    const models = modelEntities.map((e) => {
      // Map semantic model names to actual model IDs
      switch (e.value) {
        case "gradient boosting":
          return "model1"
        case "random forest":
          return "model2"
        case "xgboost":
          return "model3"
        case "neural network":
          return "model4"
        case "lightgbm":
          return "model5"
        default:
          return e.value
      }
    })
    filters.push({
      type: "model",
      value: models,
      confidence: modelEntities.reduce((acc, e) => acc + e.confidence, 0) / modelEntities.length,
    })
    modelEntities.forEach((e) => usedEntities.add(e))
  }

  // Process severity entities
  const severityEntities = entities.filter((e) => e.type === "severity" && !usedEntities.has(e))
  if (severityEntities.length > 0) {
    // Just use the highest confidence severity if multiple are found
    const highestConfidenceSeverity = severityEntities.reduce(
      (prev, current) => (current.confidence > prev.confidence ? current : prev),
      severityEntities[0],
    )
    filters.push({
      type: "severity",
      value: highestConfidenceSeverity.value,
      confidence: highestConfidenceSeverity.confidence,
    })
    severityEntities.forEach((e) => usedEntities.add(e))
  }

  // Process comparison entities
  const comparisonEntities = entities.filter((e) => e.type === "comparison" && !usedEntities.has(e))
  if (comparisonEntities.length > 0) {
    comparisonEntities.forEach((entity) => {
      filters.push({
        type: "metric",
        value: entity.value,
        confidence: entity.confidence,
      })
      usedEntities.add(entity)
    })
  }

  // Process location entities
  const locationEntities = entities.filter((e) => e.type === "location" && !usedEntities.has(e))
  if (locationEntities.length > 0) {
    locationEntities.forEach((entity) => {
      filters.push({
        type: "location",
        value: entity.value,
        confidence: entity.confidence,
      })
      usedEntities.add(entity)
    })
  }

  return filters
},

/**
 * Generate a human-readable explanation of how the query was interpreted
 */
generateQueryExplanation(result: NLPResult): string {
  const corrections: { [key: string]: string } = {
    gradent: "gradient",
    boosting: "boosting",
    randomforest: "random forest",
    xgbost: "xgboost",
    nural: "neural",
    nueral: "neural",
    netowork: "network",
    "nerual network": "neural network",
    lightgmb: "lightgbm",
    "ligt gbm": "light gbm",
    sucess: "success",
    successfull: "successful",
    sucessful: "successful",
    fail: "failed",
    failing: "failed",
    error: "errors",
    warinng: "warning",
    warnin: "warning",
    runing: "running",
    runnig: "running",
    pednig: "pending",
    critcal: "critical",
    sever: "severe",
    infomration: "information",
    accurcay: "accuracy",
    precsion: "precision",
    recal: "recall",
    yestday: "yesterday",
    yesderday: "yesterday",
    tod: "today",
    lst: "last",
    previus: "previous",
  }

  let correctedQuery = query.toLowerCase()
  let madeCorrection = false

  // Check for exact corrections
  Object.entries(corrections).forEach(([typo, correction]) => {
    if (correctedQuery.includes(typo)) {
      correctedQuery = correctedQuery.replace(new RegExp(`\\b${typo}\\b`, "gi"), correction)
      madeCorrection = true
    }
  })

  return madeCorrection ? correctedQuery : null
}

/**
 * Functions for NLP processing
 */
const NLPService = {
  /**
   * Process feedback to improve NLP understanding
   */
  processUserFeedback(originalQuery: string, correctedQuery: string): void {
    // In a production environment, this would send feedback to a learning system
    console.log(`User feedback: "${originalQuery}" should be interpreted as "${correctedQuery}"`);
    // Store feedback for future improvements
  },

  /**
   * Convert NLP result to dashboard filters
   */
  convertNLPResultToFilters(result: NLPResult): Record<string, any> {
  const intents = classifyIntent(correctedQuery || query, entities)

  // Convert entities to filter parameters
  const filters = entitiesToFilters(entities)

  // Calculate overall confidence
  const confidence = intents.length > 0 ? intents[0].confidence : 0.5

  // Identify missing information
  const missingInformation: { type: string; message: string }[] = []

  // Check if we have temporal information for time-related queries
  if (intents.some((i) => i.type === "trend") && !filters.some((f) => f.type === "date")) {
    missingInformation.push({
      type: "date",
      message: "Time period not specified for trend analysis. Consider adding a date range.",
    })
  }

  // Check if comparison operators have metric values
  if (
    entities.some((e) => e.type === "comparison") &&
    !entities.some((e) => e.type === "metric" || (e.type === "comparison" && e.value.metric))
  ) {
    missingInformation.push({
      type: "metric",
      message: "Comparison operator found but no specific metric mentioned.",
    })
  }

  return {
    query,
    normalizedQuery,
    intents,
    entities,
    filters,
    confidence,
    missingInformation: missingInformation.length > 0 ? missingInformation : undefined,
    correctedQuery: correctedQuery || undefined,
  }
}

/**
 * Generate more intelligent query suggestions based on partial query and context
 */
export function generateIntelligentSuggestions(
  partialQuery: string,
  recentQueries: string[] = [],
  options: any,
  currentPage: string
): string[] {
  const suggestions: string[] = []

  // Process the partial query with NLP to understand its intent and entities
  const nlpResult = processQueryWithNLP(partialQuery)

  // If the query is very short, provide general suggestions based on dashboard context
  if (partialQuery.length < 3) {
    if (currentPage === "model-performance") {
      return [
        "Show models with accuracy above 0.9",
        "Find failed model runs from last week",
        "Compare XGBoost and Neural Network models",
        "Show trend of model accuracy over last month",
        "Find anomalies in model performance",
      ]
    } else if (currentPage === "data-quality") {
      return [
        "Show critical data quality issues",
        "Find datasets with missing values",
        "Show trend of data quality score",
        "Compare data quality across sources",
        "Show warnings from last 3 days",
      ]
    } else {
      return [
        "Show failed runs from last week",
        "Find models with high accuracy",
        "Show critical alerts",
        "Show Neural Network models",
        "Show trend of pipeline health",
      ]
    }
  }

  // Based on identified intent, suggest completions
  if (nlpResult.intents.length > 0) {
    const primaryIntent = nlpResult.intents[0].type

    if (primaryIntent === "filter") {
      if (nlpResult.entities.some((e) => e.type === "model")) {
        suggestions.push(
          "Show gradient boosting models with accuracy above 0.9",
          "Show failed runs for neural network models",
          "Show model performance for XGBoost models from last month",
        )
      } else if (nlpResult.entities.some((e) => e.type === "status")) {
        suggestions.push(
          "Show failed runs from last week",
          "Show successful runs with high accuracy",
          "Show warning alerts from neural network models",
        )
      } else if (nlpResult.entities.some((e) => e.type === "date")) {
        suggestions.push(
          "Show critical alerts from last week",
          "Show model performance trend from last month",
          "Show pipeline runs from yesterday with warnings",
        )
      } else {
        suggestions.push(
          "Show critical alerts",
          "Show models with high accuracy",
          "Show runs from last week",
          "Show pipeline health trends",
          "Show data quality issues",
        )
      }
    }

  // If we have entity information but no clear intent, suggest based on entities
  if (nlpResult.intents[0].confidence < 0.7 && nlpResult.entities.length > 0) {
    const entityTypes = new Set(nlpResult.entities.map((e) => e.type))

    if (entityTypes.has("model")) {
      suggestions.push(
        "Show Neural Network models from last week",
        "Compare Gradient Boosting and XGBoost models",
        "Show accuracy for LightGBM models",
      )
    }

    if (entityTypes.has("status")) {
      suggestions.push(
        "Show failed runs from last week",
        "Count successful vs failed runs",
        "Show trends for warning status alerts",
      )
    }
  }

  // Add suggestions based on recent queries if relevant
  if (recentQueries.length > 0) {
    // Find recent queries that might be relevant to the current partial query
    const relevantRecentQueries = recentQueries
      .filter((q) => q.toLowerCase().includes(partialQuery.toLowerCase()))
      .slice(0, 2)

    suggestions.push(...relevantRecentQueries)
  }

  // If we still don't have enough suggestions, add some generic ones
  if (suggestions.length < 5) {
    suggestions.push(
      "Show critical alerts from last week",
      "Find models with accuracy above 0.9",
      "Show failed pipeline runs",
      "Compare model performance",
      "Show data quality trends",
    )
  }

  // Return a maximum of 5 suggestions without duplicates
  return [...new Set(suggestions)].slice(0, 5)
}

/**
 * Process feedback to improve NLP understanding
 */
export function processUserFeedback(originalQuery: string, correctedQuery: string): void {
  // In a production environment, this would send feedback to a learning system
  console.log(`User feedback: "${originalQuery}" should be interpreted as "${correctedQuery}"`);
  // Store feedback for future improvements
}

/**
 * Convert NLP result to dashboard filters
 */
export function convertNLPResultToFilters(result: NLPResult): Record<string, any> {
  const filters: Record<string, any> = {};

  // Store the original query for reference
  filters._originalQuery = result.query

  // Process each filter based on type
  result.filters.forEach((filter) => {
    if (filter.confidence >= 0.6) {
      filters[filter.type] = filter.value
    }
  })

  // Add sort options if available
  if (result.sortOptions) {
    filters._sort = result.sortOptions
  }

  // Add the primary intent
  if (result.intents.length > 0) {
    filters._intent = result.intents[0].type
  }

  return filters
}

/**
 * Generate a human-readable explanation of how the query was interpreted
 */
export function generateQueryExplanation(result: NLPResult): string {
  let explanation = ""

  // If we corrected the query, mention it
  if (result.correctedQuery) {
    explanation += `I assumed you meant: "${result.correctedQuery}"\n\n`
  }

  // Explain the interpreted intent
  if (result.intents.length > 0) {
    const primaryIntent = result.intents[0]

    switch (primaryIntent.type) {
      case "filter":
        explanation += "I understood you want to filter the dashboard data"
        break
      case "sort":
        explanation += "I understood you want to sort the dashboard data"
        break
      case "compare":
        explanation += "I understood you want to compare different items"
        break
      case "trend":
        explanation += "I understood you want to see trends over time"
        break
      case "summarize":
        explanation += "I understood you want a summary of the data"
        break
      case "count":
        explanation += "I understood you want to count or aggregate items"
        break
      case "find_anomalies":
        explanation += "I understood you want to find unusual patterns or outliers"
        break
      default:
        explanation += "I understood your request as follows"
    }

    // Add confidence level indicator for transparent communication
    if (primaryIntent.confidence < 0.7) {
      explanation += " (low confidence)"
    }

    explanation += ":\n\n"
  }

  // Explain the filters that were applied
  const filterDescriptions = []

  result.filters.forEach((filter) => {
    switch (filter.type) {
      case "date":
        const from = filter.value.from ? new Date(filter.value.from).toLocaleDateString() : undefined
        const to = filter.value.to ? new Date(filter.value.to).toLocaleDateString() : undefined

        if (from && to) {
          filterDescriptions.push(`Date range: ${from} to ${to}`)
        } else if (from) {
          filterDescriptions.push(`Date from: ${from}`)
        } else if (to) {
          filterDescriptions.push(`Date to: ${to}`)
        }
        break

      case "status":
        const statuses = Array.isArray(filter.value) ? filter.value.join(", ") : filter.value
        filterDescriptions.push(`Status: ${statuses}`)
        break

      case "model":
        const models = Array.isArray(filter.value) ? filter.value.join(", ") : filter.value
        filterDescriptions.push(`Models: ${models}`)
        break

      case "severity":
        filterDescriptions.push(`Severity: ${filter.value}`)
        break

      case "metric":
        const metric = filter.value
        if (metric.operator && metric.metric) {
          filterDescriptions.push(`${metric.metric} ${metric.operator} ${metric.value}`)
        }
        break

      case "search":
        filterDescriptions.push(`Search: "${filter.value}"`)
        break
    }
  })

  // Add sort information if available
  if (result.sortOptions) {
    const direction = result.sortOptions.direction === "desc" ? "descending" : "ascending"
    filterDescriptions.push(`Sorting by ${result.sortOptions.field} (${direction})`)
  }

  // Add the filter descriptions to the explanation
  if (filterDescriptions.length > 0) {
    explanation += `• ${filterDescriptions.join("\n• ")}`
  } else {
    explanation += "I couldn't extract specific filters from your query."
  }

  // Add information about missing information if any
  if (result.missingInformation && result.missingInformation.length > 0) {
    explanation += "\n\nAdditional information that would help:\n"
    result.missingInformation.forEach((missing) => {
      explanation += `• ${missing.message}\n`
    })
  }

  return explanation
}

/**
 * Convert a search query to a full natural language query
 * This helps transform simple keyword searches into proper natural language
 */
export function expandKeywordQuery(query: string): string {
  // If the query already looks like natural language, return it unchanged
  if (query.length > 20 || query.includes(" ")) {
    return query
  }

  // Detect what type of keyword this might be
  if (/model\d+|gradient|random|forest|xgboost|neural|network|lightgbm/i.test(query)) {
    return `Show ${query} models`
  }

  if (/success|fail|error|warning|running|pending/i.test(query)) {
    return `Show ${query} runs`
  }

  if (/critical|severe|info/i.test(query)) {
    return `Show ${query} alerts`
  }

  if (/accuracy|precision|recall|f1|auc|rmse|mae/i.test(query)) {
    return `Show models sorted by ${query}`
  }

  if (/today|yesterday|week|month/i.test(query)) {
    return `Show data from ${query}`
  }

  // Default expansion
  return `Search for ${query}`
},

/**
 * Process query using advanced NLP techniques
 */
processQueryWithNLP(query: string): NLPResult {
  const normalizedQuery = query.toLowerCase().trim()

  // Try to correct the query if it contains typos
  const correctedQuery = correctQuery(query)

  // Extract entities from the query (or corrected query if available)
  const entities = extractEntities(correctedQuery || query)

  // Classify the intent of the query
  const intents = classifyIntent(correctedQuery || query, entities)

  // Convert entities to filter parameters
  const filters = entitiesToFilters(entities)

  // Calculate overall confidence
  const confidence = intents.length > 0 ? intents[0].confidence : 0.5

  // Identify missing information
  const missingInformation: { type: string; message: string }[] = []

  // Check if we have temporal information for time-related queries
  if (intents.some((i) => i.type === "trend") && !filters.some((f) => f.type === "date")) {
    missingInformation.push({
      type: "date",
      message: "Time period not specified for trend analysis. Consider adding a date range.",
    })
  }

  // Check if comparison operators have metric values
  if (
    entities.some((e) => e.type === "comparison") &&
    !entities.some((e) => e.type === "metric" || (e.type === "comparison" && e.value.metric))
  ) {
    missingInformation.push({
      type: "metric",
      message: "Comparison operator found but no specific metric mentioned.",
    })
  }

  return {
    query,
    normalizedQuery,
    intents,
    entities,
    filters,
    confidence,
    missingInformation: missingInformation.length > 0 ? missingInformation : undefined,
    correctedQuery: correctedQuery || undefined,
  }
},

/**
 * Generate more intelligent query suggestions based on partial query and context
 */
generateIntelligentSuggestions(
  partialQuery: string,
  recentQueries: string[] = [],
  options: any,
  currentPage: string
): string[] {
  const suggestions: string[] = []

  // Process the partial query with NLP to understand its intent and entities
  const nlpResult = this.processQueryWithNLP(partialQuery)

  // If the query is very short, provide general suggestions based on dashboard context
  if (partialQuery.length < 3) {
    if (currentPage === "model-performance") {
      return [
        "Show models with accuracy above 0.9",
        "Find failed model runs from last week",
        "Compare XGBoost and Neural Network models",
        "Show trend of model accuracy over last month",
        "Find anomalies in model performance",
      ]
    } else if (currentPage === "data-quality") {
      return [
        "Show critical data quality issues",
        "Find datasets with missing values",
        "Show trend of data quality score",
        "Compare data quality across sources",
        "Show warnings from last 3 days",
      ]
    } else {
      return [
        "Show failed runs from last week",
        "Find models with high accuracy",
        "Show critical alerts",
        "Show Neural Network models",
        "Show trend of pipeline health",
      ]
    }
  }

  // Based on identified intent, suggest completions
  if (nlpResult.intents.length > 0) {
    const primaryIntent = nlpResult.intents[0].type

    if (primaryIntent === "filter") {
      if (nlpResult.entities.some((e) => e.type === "model")) {
        suggestions.push(
          "Show gradient boosting models with accuracy above 0.9",
          "Show failed runs for neural network models",
          "Show model performance for XGBoost models from last month",
        )
      } else if (nlpResult.entities.some((e) => e.type === "status")) {
        suggestions.push(
          "Show failed runs from last week",
          "Show successful runs with high accuracy",
          "Show warning alerts from neural network models",
        )
      } else if (nlpResult.entities.some((e) => e.type === "date")) {
        suggestions.push(
          "Show critical alerts from last week",
          "Show model performance trend from last month",
          "Show pipeline runs from yesterday with warnings",
        )
      } else {
        suggestions.push(
          "Show critical alerts",
          "Show models with high accuracy",
          "Show runs from last week",
          "Show pipeline health trends",
          "Show data quality issues",
        )
      }
    }
  }

  // If we have entity information but no clear intent, suggest based on entities
  if (nlpResult.intents[0].confidence < 0.7 && nlpResult.entities.length > 0) {
    const entityTypes = new Set(nlpResult.entities.map((e) => e.type))

    if (entityTypes.has("model")) {
      suggestions.push(
        "Show Neural Network models from last week",
        "Compare Gradient Boosting and XGBoost models",
        "Show accuracy for LightGBM models",
      )
    }

    if (entityTypes.has("status")) {
      suggestions.push(
        "Show failed runs from last week",
        "Count successful vs failed runs",
        "Show trends for warning status alerts",
      )
    }
  }

  // Add suggestions based on recent queries if relevant
  if (recentQueries.length > 0) {
    // Find recent queries that might be relevant to the current partial query
    const relevantRecentQueries = recentQueries
      .filter((q) => q.toLowerCase().includes(partialQuery.toLowerCase()))
      .slice(0, 2)

    suggestions.push(...relevantRecentQueries)
  }

  // If we still don't have enough suggestions, add some generic ones
  if (suggestions.length < 5) {
    suggestions.push(
      "Show critical alerts from last week",
      "Find models with accuracy above 0.9",
      "Show failed pipeline runs",
      "Compare model performance",
      "Show data quality trends",
    )
  }

  // Return a maximum of 5 suggestions without duplicates
  return [...new Set(suggestions)].slice(0, 5)
}
};

// Move the previously exported functions to be internal helper functions
/**
 * Finds semantically related terms in the query
 */
function findSemanticTerms(query: string): Map<string, Entity[]> {
  const normalizedQuery = query.toLowerCase()
  const result = new Map<string, Entity[]>()

  // For each term in our vocabulary
  Object.entries(semanticVocabulary).forEach(([key, termData]) => {
    const allTerms = [termData.term, ...termData.synonyms]
    const foundEntities: Entity[] = []

    // Check if any of the terms or synonyms are in the query
    allTerms.forEach((term) => {
      const regex = new RegExp(`\\b${term}\\b`, "gi")
      let match
      while ((match = regex.exec(normalizedQuery)) !== null) {
        foundEntities.push({
          text: match[0],
          type: termData.category as EntityType,
          startPosition: match.index,
          endPosition: match.index + match[0].length,
          value: key, // Use the canonical term as the value
          confidence: term === termData.term ? 0.95 : 0.85, // Higher confidence for exact matches
        })
      }
    })

    // Add the found entities if any
    if (foundEntities.length > 0) {
      result.set(key, foundEntities)
    }
  })

  return result
}

/**
 * Identify named entities in the query
 */
function extractEntities(query: string): Entity[] {
  const entities: Entity[] = []
  const normalizedQuery = query.toLowerCase()

  // Extract semantic terms first
  const semanticTerms = findSemanticTerms(query)
  semanticTerms.forEach((termEntities) => {
    entities.push(...termEntities)
  })

  // Extract dates with regex patterns
  const datePatterns = [
    {
      regex: /\b(?:from|since|after)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b/i,
      type: "date" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        const dateStr = match[1]
        try {
          return { from: new Date(dateStr), to: null }
        } catch (e) {
          return { text: dateStr }
        }
      },
    },
    {
      regex: /\b(?:to|until|before)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b/i,
      type: "date" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        const dateStr = match[1]
        try {
          return { from: null, to: new Date(dateStr) }
        } catch (e) {
          return { text: dateStr }
        }
      },
    },
    {
      regex: /\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+to\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b/i,
      type: "date" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        try {
          return { from: new Date(match[1]), to: new Date(match[2]) }
        } catch (e) {
          return { text: `${match[1]} to ${match[2]}` }
        }
      },
    },
    {
      regex: /\blast\s+(\d+)Ñ+(day|days|week|weeks|month|months|year|years)\b/i,
      type: "date" as EntityType,
      valueExtractor: (match: RegExpExecArray) => {
        const num = Number.parseInt(match[1], 10)
        const unit = match[2].toLowerCase()
        const to = new Date()
        const from = new Date()

        
