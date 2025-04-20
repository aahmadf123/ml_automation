"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  type TooltipProps,
} from "recharts"
import { ChartContainer } from "@/components/ui/chart"

// Generate mock data for feature distribution
const generateMockDistribution = (feature: string) => {
  const baseData = []
  const mean = Math.random() * 0.5 + 0.25
  const stdDev = Math.random() * 0.15 + 0.05

  for (let i = 0; i < 100; i++) {
    // Generate random values with normal-ish distribution
    const x = Math.random()
    const y = Math.exp(-Math.pow((x - mean) / stdDev, 2) / 2) / (stdDev * Math.sqrt(2 * Math.PI))
    const normalized = y / 2.5 // Normalize to get values between 0-1

    baseData.push({
      value: x,
      density: normalized,
      impact: (x - 0.5) * (Math.random() > 0.5 ? 1 : -1) * Math.random(),
      isSelected: false,
    })
  }

  // Add the selected point
  const selectedValue = Math.random()
  const selectedImpact = (selectedValue - 0.5) * (Math.random() > 0.5 ? 1 : -1) * Math.random() * 2

  baseData.push({
    value: selectedValue,
    density: 0.5,
    impact: selectedImpact,
    isSelected: true,
  })

  return baseData
}

// Custom tooltip for the force plot
const ForceTooltip = ({ active, payload }: TooltipProps<number, string>) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-background border rounded-md shadow-md p-2 text-sm">
        <p className="font-medium">{data.isSelected ? "Selected Prediction" : "Other Data Point"}</p>
        <p>Value: {data.value.toFixed(2)}</p>
        <p>
          Impact:{" "}
          <span className={data.impact > 0 ? "text-green-500" : "text-red-500"}>
            {data.impact > 0 ? "+" : ""}
            {data.impact.toFixed(3)}
          </span>
        </p>
      </div>
    )
  }
  return null
}

interface ShapForceValuePlotProps {
  feature: string
  modelId?: string
  predictionId?: string
}

export function ShapForceValuePlot({ feature, modelId = "model1", predictionId }: ShapForceValuePlotProps) {
  const [loading, setLoading] = useState(true)
  const [distributionData, setDistributionData] = useState<any[]>([])

  useEffect(() => {
    // Simulate API call to get feature distribution data
    const fetchDistributionData = async () => {
      // In a real implementation, this would fetch from your API
      // const response = await fetch(`/api/shap/distribution?modelId=${modelId}&predictionId=${predictionId}&feature=${feature}`);
      // const data = await response.json();

      // Generate mock data
      const mockData = generateMockDistribution(feature)

      // Simulate loading delay
      setTimeout(() => {
        setDistributionData(mockData)
        setLoading(false)
      }, 800)
    }

    fetchDistributionData()
  }, [feature, modelId, predictionId])

  // Find the selected point
  const selectedPoint = distributionData.find((d) => d.isSelected)

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Feature Impact: {feature}</CardTitle>
        <CardDescription>
          How this feature's value affects the prediction compared to the distribution of values
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-[300px] w-full" />
        ) : (
          <div className="space-y-2">
            <div className="text-sm text-muted-foreground">
              This chart shows how different values of {feature} impact predictions. The highlighted point shows your
              selected prediction's value.
            </div>
            <ChartContainer className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="value"
                    name="Value"
                    domain={[0, 1]}
                    label={{ value: `${feature} Value`, position: "bottom", offset: 0 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="impact"
                    name="Impact"
                    label={{ value: "Impact on Prediction", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip content={<ForceTooltip />} />
                  <ReferenceLine y={0} stroke="#666" />
                  <Scatter
                    name="Distribution"
                    data={distributionData.filter((d) => !d.isSelected)}
                    fill="#8884d8"
                    opacity={0.5}
                  />
                  {selectedPoint && (
                    <Scatter
                      name="Selected Point"
                      data={[selectedPoint]}
                      fill={selectedPoint.impact > 0 ? "#10b981" : "#ef4444"}
                      shape="circle"
                    />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            </ChartContainer>
            {selectedPoint && (
              <div className="mt-4 p-4 border rounded-md bg-muted/30">
                <p className="font-medium">Interpretation:</p>
                <p className="text-sm mt-1">
                  For this prediction, the value of <span className="font-semibold">{feature}</span> is{" "}
                  <span className="font-semibold">{selectedPoint.value.toFixed(2)}</span>, which has a{" "}
                  <span
                    className={selectedPoint.impact > 0 ? "text-green-500 font-semibold" : "text-red-500 font-semibold"}
                  >
                    {selectedPoint.impact > 0 ? "positive" : "negative"} impact
                  </span>{" "}
                  on the prediction.
                  {selectedPoint.impact > 0
                    ? " This increases the probability of loss."
                    : " This decreases the probability of loss."}
                </p>
                <p className="text-sm mt-2">
                  {selectedPoint.value > 0.5
                    ? `This value is higher than average for ${feature}.`
                    : `This value is lower than average for ${feature}.`}
                  {Math.abs(selectedPoint.impact) > 0.1
                    ? " The impact is significant compared to other features."
                    : " The impact is relatively small compared to other features."}
                </p>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
