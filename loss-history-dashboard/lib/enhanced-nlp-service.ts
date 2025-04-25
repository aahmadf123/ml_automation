"use client";

// Basic types needed by the natural-language query component
export interface Entity {
  text: string;
  type: string;
  startPosition: number;
  endPosition: number;
  value: any;
  confidence: number;
}

export interface Intent {
  type: string;
  confidence: number;
  entities: Entity[];
}

export interface NLPResult {
  query: string;
  normalizedQuery: string;
  intents: Intent[];
  entities: Entity[];
  filters: any[];
  confidence: number;
  correctedQuery?: string;
}

/**
 * Process query using NLP techniques - simplified implementation
 */
export function processQueryWithNLP(query: string): NLPResult {
  return {
    query,
    normalizedQuery: query.toLowerCase(),
    intents: [],
    entities: [],
    filters: [],
    confidence: 0.8
  };
}

/**
 * Convert NLP result to dashboard filters
 */
export function convertNLPResultToFilters(result: NLPResult): Record<string, any> {
  return {
    _originalQuery: result.query,
    // Add default filters if needed
    status: ["all"],
    timeRange: "week"
  };
}

/**
 * Generate intelligent query suggestions
 */
export function generateIntelligentSuggestions(
  partialQuery: string,
  recentQueries: string[] = [],
  options: any = {},
  currentPage: string = "dashboard"
): string[] {
  // Default suggestions
  return [
    "Show models with accuracy above 0.9",
    "Find failed model runs from last week",
    "Compare Model1 and Model4",
    "Show trend of model accuracy over last month",
    "Find anomalies in model performance"
  ];
}

/**
 * Generate query explanation
 */
export function generateQueryExplanation(result: NLPResult): string {
  return `I processed your query: "${result.query}"`;
}

/**
 * Process user feedback
 */
export function processUserFeedback(originalQuery: string, correctedQuery: string): void {
  console.log(`User feedback: "${originalQuery}" should be interpreted as "${correctedQuery}"`);
}

/**
 * Convert keyword query to natural language
 */
export function expandKeywordQuery(query: string): string {
  if (query.length < 5 || !query.includes(" ")) {
    return `Show ${query} models`;
  }
  return query;
}

