"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ArrowUp, ArrowDown, Minus } from "lucide-react"

// Mock feature ranking data
const mockFeatureRankings = {
  model1: [
    { feature: "ClaimAmount", rank: 1, importance: 0.22 },
    { feature: "PropertyAge", rank: 2, importance: 0.18 },
    { feature: "ClaimHistory", rank: 3, importance: 0.15 },
    { feature: "LocationRiskScore", rank: 4, importance: 0.12 },
    { feature: "PropertyValue", rank: 5, importance: 0.09 },
    { feature: "CoverageLevel", rank: 6, importance: 0.07 },
    { feature: "ClaimType", rank: 7, importance: 0.06 },
    { feature: "WeatherEvents", rank: 8, importance: 0.04 },
    { feature: "SecuritySystem", rank: 9, importance: 0.03 },
    { feature: "OwnershipDuration", rank: 10, importance: 0.02 },
  ],
  model2: [
    { feature: "ClaimAmount", rank: 1, importance: 0.25 },
    { feature: "ClaimHistory", rank: 2, importance: 0.2 },
    { feature: "PropertyAge", rank: 3, importance: 0.15 },
    { feature: "LocationRiskScore", rank: 4, importance: 0.1 },
    { feature: "ClaimType", rank: 5, importance: 0.08 },
    { feature: "PropertyValue", rank: 6, importance: 0.07 },
    { feature: "CoverageLevel", rank: 7, importance: 0.06 },
    { feature: "WeatherEvents", rank: 8, importance: 0.04 },
    { feature: "OwnershipDuration", rank: 9, importance: 0.03 },
    { feature: "SecuritySystem", rank: 10, importance: 0.02 },
  ],
  model3: [
    { feature: "ClaimHistory", rank: 1, importance: 0.24 },
    { feature: "ClaimAmount", rank: 2, importance: 0.2 },
    { feature: "LocationRiskScore", rank: 3, importance: 0.16 },
    { feature: "PropertyAge", rank: 4, importance: 0.12 },
    { feature: "ClaimType", rank: 5, importance: 0.09 },
    { feature: "WeatherEvents", rank: 6, importance: 0.07 },
    { feature: "PropertyValue", rank: 7, importance: 0.05 },
    { feature: "CoverageLevel", rank: 8, importance: 0.04 },
    { feature: "SecuritySystem", rank: 9, importance: 0.02 },
    { feature: "OwnershipDuration", rank: 10, importance: 0.01 },
  ],
}

// Prepare data for comparison table
const prepareRankingComparisonData = (modelData: Record<string, any[]>) => {
  const features = new Set<string>()

  // Collect all unique features
  Object.values(modelData).forEach((modelFeatures) => {
    modelFeatures.forEach((item) => features.add(item.feature))
  })

  // Create comparison data
  return Array.from(features)
    .map((feature) => {
      const result: Record<string, any> = { feature }

      Object.entries(modelData).forEach(([modelId, modelFeatures]) => {
        const featureData = modelFeatures.find((item) => item.feature === feature)
        if (featureData) {
          result[`${modelId}_rank`] = featureData.rank
          result[`${modelId}_importance`] = featureData.importance
        }
      })

      return result
    })
    .sort((a, b) => {
      // Sort by the average rank across models
      const aRanks = Object.entries(a)
        .filter(([key]) => key.endsWith("_rank"))
        .map(([_, value]) => value as number)

      const bRanks = Object.entries(b)
        .filter(([key]) => key.endsWith("_rank"))
        .map(([_, value]) => value as number)

      const aAvg = aRanks.reduce((sum, rank) => sum + rank, 0) / aRanks.length
      const bAvg = bRanks.reduce((sum, rank) => sum + rank, 0) / bRanks.length

      return aAvg - bAvg
    })
}

interface FeatureRankingComparisonProps {
  modelIds: string[]
  modelNames?: Record<string, string>
}

export function FeatureRankingComparison({
  modelIds,
  modelNames = {
    model1: "Loss Prediction",
    model2: "Claim Amount",
    model3: "Fraud Detection",
  },
}: FeatureRankingComparisonProps) {
  const [loading, setLoading] = useState(true)
  const [comparisonData, setComparisonData] = useState<any[]>([])
  const [modelData, setModelData] = useState<Record<string, any[]>>({})

  useEffect(() => {
    // Simulate API call to get feature rankings for multiple models
    const fetchFeatureRankings = async () => {
      // In a real implementation, this would fetch from your API
      // const promises = modelIds.map(id => fetch(`/api/feature-rankings?modelId=${id}`).then(res => res.json()))
      // const results = await Promise.all(promises)

      // Simulate loading delay
      setTimeout(() => {
        // Filter mock data to only include requested models
        const filteredData = Object.fromEntries(
          Object.entries(mockFeatureRankings).filter(([key]) => modelIds.includes(key)),
        )

        setModelData(filteredData)
        setComparisonData(prepareRankingComparisonData(filteredData))
        setLoading(false)
      }, 1000)
    }

    if (modelIds.length > 0) {
      fetchFeatureRankings()
    }
  }, [modelIds])

  // Calculate rank difference between models
  const getRankDifference = (feature: string, modelId1: string, modelId2: string) => {
    const featureData = comparisonData.find((item) => item.feature === feature)
    if (!featureData) return 0

    const rank1 = featureData[`${modelId1}_rank`]
    const rank2 = featureData[`${modelId2}_rank`]

    if (rank1 === undefined || rank2 === undefined) return 0

    return rank2 - rank1
  }

  // Render rank difference indicator
  const renderRankDifference = (diff: number) => {
    if (diff === 0) {
      return <Minus className="h-4 w-4 text-gray-400" />
    } else if (diff > 0) {
      return (
        <div className="flex items-center text-green-500">
          <ArrowUp className="h-4 w-4 mr-1" />
          <span>{Math.abs(diff)}</span>
        </div>
      )
    } else {
      return (
        <div className="flex items-center text-red-500">
          <ArrowDown className="h-4 w-4 mr-1" />
          <span>{Math.abs(diff)}</span>
        </div>
      )
    }
  }

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle>Feature Ranking Comparison</CardTitle>
        <CardDescription>Compare how features are ranked across different models</CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-[500px] w-full" />
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[200px]">Feature</TableHead>
                  {modelIds.map((modelId) => (
                    <TableHead key={modelId} className="text-center">
                      {modelNames[modelId] || modelId}
                    </TableHead>
                  ))}
                  {modelIds.length > 1 && <TableHead className="text-center">Rank Differences</TableHead>}
                </TableRow>
              </TableHeader>
              <TableBody>
                {comparisonData.map((item) => (
                  <TableRow key={item.feature}>
                    <TableCell className="font-medium">{item.feature}</TableCell>
                    {modelIds.map((modelId) => (
                      <TableCell key={modelId} className="text-center">
                        {item[`${modelId}_rank`] !== undefined ? (
                          <div className="flex flex-col items-center">
                            <Badge variant="outline" className="mb-1">
                              Rank {item[`${modelId}_rank`]}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {(item[`${modelId}_importance`] * 100).toFixed(1)}%
                            </span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground">N/A</span>
                        )}
                      </TableCell>
                    ))}
                    {modelIds.length > 1 && (
                      <TableCell>
                        <div className="flex justify-center space-x-4">
                          {modelIds.slice(0, -1).map((modelId, index) => (
                            <div key={modelId} className="flex items-center">
                              <span className="text-xs text-muted-foreground mr-1">
                                {modelNames[modelId]?.charAt(0) || modelId} →{" "}
                                {modelNames[modelIds[index + 1]]?.charAt(0) || modelIds[index + 1]}:
                              </span>
                              {renderRankDifference(getRankDifference(item.feature, modelId, modelIds[index + 1]))}
                            </div>
                          ))}
                        </div>
                      </TableCell>
                    )}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
