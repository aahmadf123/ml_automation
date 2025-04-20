"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CheckCircle, AlertTriangle, XCircle } from "lucide-react"

interface ValidationResult {
  valid: boolean
  warnings: ValidationIssue[]
  errors: ValidationIssue[]
}

interface ValidationIssue {
  type: string
  column: string
  count: number
  percentage: number
}

interface ValidationRulesProps {
  results: ValidationResult
}

export function ValidationRules({ results }: ValidationRulesProps) {
  const [activeTab, setActiveTab] = useState("summary")

  const totalIssues = results.warnings.length + results.errors.length

  const getStatusIcon = () => {
    if (results.errors.length > 0) {
      return <XCircle className="h-8 w-8 text-red-500" />
    } else if (results.warnings.length > 0) {
      return <AlertTriangle className="h-8 w-8 text-amber-500" />
    } else {
      return <CheckCircle className="h-8 w-8 text-green-500" />
    }
  }

  const getStatusText = () => {
    if (results.errors.length > 0) {
      return "Validation Failed"
    } else if (results.warnings.length > 0) {
      return "Validation Passed with Warnings"
    } else {
      return "Validation Passed"
    }
  }

  const getStatusColor = () => {
    if (results.errors.length > 0) {
      return "text-red-500"
    } else if (results.warnings.length > 0) {
      return "text-amber-500"
    } else {
      return "text-green-500"
    }
  }

  const getIssueTypeLabel = (type: string) => {
    switch (type) {
      case "missing_values":
        return "Missing Values"
      case "outliers":
        return "Outliers"
      case "invalid_format":
        return "Invalid Format"
      case "duplicates":
        return "Duplicate Values"
      default:
        return type.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-muted/30 p-6 rounded-lg">
        <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4">
          <div className="flex-shrink-0">{getStatusIcon()}</div>
          <div className="text-center sm:text-left">
            <h3 className={`text-lg font-medium ${getStatusColor()}`}>{getStatusText()}</h3>
            <p className="text-sm text-muted-foreground">
              {totalIssues === 0
                ? "No issues found in the data"
                : `Found ${results.warnings.length} warnings and ${results.errors.length} errors`}
            </p>
          </div>
        </div>
      </div>

      <Tabs defaultValue="summary" value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="warnings" disabled={results.warnings.length === 0}>
            Warnings ({results.warnings.length})
          </TabsTrigger>
          <TabsTrigger value="errors" disabled={results.errors.length === 0}>
            Errors ({results.errors.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="font-medium">Data Quality Score</h4>
                  <Badge variant={results.valid ? "default" : "destructive"}>
                    {results.valid ? "Passed" : "Failed"}
                  </Badge>
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm">Overall Quality</span>
                      <span className="text-sm font-medium">{results.errors.length === 0 ? "Good" : "Poor"}</span>
                    </div>
                    <Progress
                      value={results.errors.length === 0 ? 100 : 50}
                      className={results.errors.length === 0 ? "bg-green-100" : "bg-red-100"}
                    />
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm">Completeness</span>
                      <span className="text-sm font-medium">
                        {results.warnings.some((w) => w.type === "missing_values") ? "Partial" : "Complete"}
                      </span>
                    </div>
                    <Progress
                      value={results.warnings.some((w) => w.type === "missing_values") ? 70 : 100}
                      className={
                        results.warnings.some((w) => w.type === "missing_values") ? "bg-amber-100" : "bg-green-100"
                      }
                    />
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm">Consistency</span>
                      <span className="text-sm font-medium">
                        {results.warnings.some((w) => w.type === "outliers") ? "Inconsistent" : "Consistent"}
                      </span>
                    </div>
                    <Progress
                      value={results.warnings.some((w) => w.type === "outliers") ? 80 : 100}
                      className={results.warnings.some((w) => w.type === "outliers") ? "bg-amber-100" : "bg-green-100"}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <h4 className="font-medium mb-4">Issues by Column</h4>
                {totalIssues === 0 ? (
                  <div className="flex items-center justify-center h-32 text-muted-foreground">
                    <div className="text-center">
                      <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-500" />
                      <p>No issues found</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {[
                      ...new Set([...results.warnings.map((w) => w.column), ...results.errors.map((e) => e.column)]),
                    ].map((column) => {
                      const columnWarnings = results.warnings.filter((w) => w.column === column)
                      const columnErrors = results.errors.filter((e) => e.column === column)
                      const hasIssues = columnWarnings.length > 0 || columnErrors.length > 0

                      return (
                        <div key={column} className="flex items-center justify-between">
                          <div className="flex items-center">
                            {columnErrors.length > 0 ? (
                              <XCircle className="h-4 w-4 text-red-500 mr-2" />
                            ) : columnWarnings.length > 0 ? (
                              <AlertTriangle className="h-4 w-4 text-amber-500 mr-2" />
                            ) : (
                              <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                            )}
                            <span>{column}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            {columnWarnings.length > 0 && (
                              <Badge variant="outline" className="bg-amber-100 text-amber-700 hover:bg-amber-100/80">
                                {columnWarnings.length} warnings
                              </Badge>
                            )}
                            {columnErrors.length > 0 && (
                              <Badge variant="outline" className="bg-red-100 text-red-700 hover:bg-red-100/80">
                                {columnErrors.length} errors
                              </Badge>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {totalIssues > 0 && (
            <div className="flex justify-end">
              <Button variant="outline" onClick={() => setActiveTab(results.errors.length > 0 ? "errors" : "warnings")}>
                View Details
              </Button>
            </div>
          )}
        </TabsContent>

        <TabsContent value="warnings" className="space-y-4 mt-4">
          {results.warnings.map((warning, index) => (
            <Card key={index}>
              <CardContent className="p-4">
                <div className="flex items-start">
                  <AlertTriangle className="h-5 w-5 text-amber-500 mr-3 mt-0.5" />
                  <div>
                    <h4 className="font-medium">
                      {getIssueTypeLabel(warning.type)} in column "{warning.column}"
                    </h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Found {warning.count} {warning.count === 1 ? "instance" : "instances"} (
                      {(warning.percentage * 100).toFixed(1)}% of data)
                    </p>
                    <div className="mt-2 text-sm">
                      <p>Recommendation: {getRecommendation(warning.type)}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="errors" className="space-y-4 mt-4">
          {results.errors.map((error, index) => (
            <Card key={index}>
              <CardContent className="p-4">
                <div className="flex items-start">
                  <XCircle className="h-5 w-5 text-red-500 mr-3 mt-0.5" />
                  <div>
                    <h4 className="font-medium">
                      {getIssueTypeLabel(error.type)} in column "{error.column}"
                    </h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Found {error.count} {error.count === 1 ? "instance" : "instances"} (
                      {(error.percentage * 100).toFixed(1)}% of data)
                    </p>
                    <div className="mt-2 text-sm">
                      <p>Recommendation: {getRecommendation(error.type)}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  )
}

function getRecommendation(issueType: string): string {
  switch (issueType) {
    case "missing_values":
      return "Consider imputing missing values using mean, median, or mode based on data distribution."
    case "outliers":
      return "Review outliers to determine if they are valid data points or errors that need correction."
    case "invalid_format":
      return "Standardize the format of values in this column to ensure consistency."
    case "duplicates":
      return "Remove or merge duplicate entries to maintain data integrity."
    default:
      return "Review and clean the data according to your business rules."
  }
}
