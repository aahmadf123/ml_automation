"use client"

import { useState, useMemo, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { ChartContainer } from "@/components/ui/chart"
import { Badge } from "@/components/ui/badge"
import {
  BarChart,
  LineChart,
  ScatterChartIcon,
  PieChart,
  AreaChart,
  Download,
  Save,
  Wand2,
  RefreshCw,
  Plus,
  Trash2,
  X,
  Pencil,
  MapPin,
} from "lucide-react"
import {
  ResponsiveContainer,
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  BarChart as RechartsBarChart,
  Bar,
  ScatterChart,
  Scatter,
  ZAxis,
  AreaChart as RechartsAreaChart,
  Area,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  ReferenceDot,
  ReferenceLabel,
  type TooltipProps,
} from "recharts"
import { Tooltip as UITooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"

// Sample data for visualization
const sampleData = [
  { month: "Jan", sales: 1000, profit: 500, customers: 100, category: "Electronics" },
  { month: "Feb", sales: 1500, profit: 700, customers: 150, category: "Electronics" },
  { month: "Mar", sales: 1200, profit: 600, customers: 120, category: "Electronics" },
  { month: "Apr", sales: 1800, profit: 900, customers: 180, category: "Clothing" },
  { month: "May", sales: 2000, profit: 1000, customers: 200, category: "Clothing" },
  { month: "Jun", sales: 2200, profit: 1100, customers: 220, category: "Clothing" },
  { month: "Jul", sales: 1900, profit: 950, customers: 190, category: "Home" },
  { month: "Aug", sales: 2100, profit: 1050, customers: 210, category: "Home" },
  { month: "Sep", sales: 2300, profit: 1150, customers: 230, category: "Electronics" },
  { month: "Oct", sales: 2500, profit: 1250, customers: 250, category: "Electronics" },
  { month: "Nov", sales: 2700, profit: 1350, customers: 270, category: "Home" },
  { month: "Dec", sales: 3000, profit: 1500, customers: 300, category: "Clothing" },
]

// Define annotation interface
interface Annotation {
  id: string
  xValue: any // The x-axis value where the annotation should be placed
  yValue: any // The y-axis value where the annotation should be placed
  label: string // The text to display
  color: string // The color of the annotation
  position: "top" | "bottom" | "left" | "right" // Position of the label relative to the point
  important: boolean // Whether this is a high-priority annotation
}

interface DataVisualizationProps {
  data?: any[]
}

export function DataVisualization({ data = sampleData }: DataVisualizationProps) {
  const [chartType, setChartType] = useState<"bar" | "line" | "scatter" | "pie" | "area" | "heatmap">("bar")
  const [xAxis, setXAxis] = useState<string>("")
  const [yAxis, setYAxis] = useState<string>("")
  const [secondaryYAxis, setSecondaryYAxis] = useState<string>("")
  const [colorBy, setColorBy] = useState<string>("")
  const [showGrid, setShowGrid] = useState<boolean>(true)
  const [showLegend, setShowLegend] = useState<boolean>(true)
  const [chartTitle, setChartTitle] = useState<string>("Data Visualization")
  const [colorScheme, setColorScheme] = useState<string>("neon")
  const [isGeneratingInsights, setIsGeneratingInsights] = useState<boolean>(false)
  const [insights, setInsights] = useState<string[]>([])
  const [aggregationType, setAggregationType] = useState<"none" | "sum" | "average" | "count">("none")
  const [binCount, setBinCount] = useState<number>(10)

  // Annotation states
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [showAnnotations, setShowAnnotations] = useState<boolean>(true)
  const [editingAnnotation, setEditingAnnotation] = useState<Annotation | null>(null)
  const [newAnnotation, setNewAnnotation] = useState<Partial<Annotation>>({
    color: "#FF6B6B",
    position: "top",
    important: false,
  })
  const [isAddingAnnotation, setIsAddingAnnotation] = useState<boolean>(false)
  const [selectedDataPoint, setSelectedDataPoint] = useState<any>(null)
  const chartRef = useRef<HTMLDivElement>(null)

  // Get all columns from the data
  const columns = useMemo(() => {
    if (!data || data.length === 0) return []
    return Object.keys(data[0])
  }, [data])

  // Determine column types (numeric, categorical, date)
  const columnTypes = useMemo(() => {
    if (!data || data.length === 0) return {}

    const types: Record<string, string> = {}

    columns.forEach((column) => {
      // Check first non-null value
      const sampleValue = data.find((row) => row[column] !== null && row[column] !== undefined)?.[column]

      if (sampleValue === undefined) {
        types[column] = "unknown"
      } else if (typeof sampleValue === "number") {
        types[column] = "numeric"
      } else if (
        !isNaN(Date.parse(sampleValue)) &&
        typeof sampleValue === "string" &&
        (sampleValue.includes("-") || sampleValue.includes("/"))
      ) {
        types[column] = "date"
      } else {
        types[column] = "categorical"
      }
    })

    return types
  }, [data, columns])

  // Set default axes based on column types when columns change
  useMemo(() => {
    if (columns.length > 0) {
      // Find first categorical column for x-axis
      const firstCategorical = columns.find((col) => columnTypes[col] === "categorical") || columns[0]

      // Find first numeric column for y-axis
      const firstNumeric = columns.find((col) => columnTypes[col] === "numeric")

      // Find second numeric column for secondary y-axis
      const secondNumeric = columns.filter((col) => columnTypes[col] === "numeric" && col !== firstNumeric)[0]

      if (firstCategorical) setXAxis(firstCategorical)
      if (firstNumeric) setYAxis(firstNumeric)
      if (secondNumeric) setSecondaryYAxis(secondNumeric)
    }
  }, [columns, columnTypes])

  // Color schemes
  const colorSchemes: Record<string, string[]> = {
    neon: ["#4ECDC4", "#FF6B6B", "#FFE66D", "#1A535C", "#F7FFF7"],
    blues: ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600"],
    greens: ["#4ECDC4", "#1A535C", "#3AAFA9", "#2B7A78", "#DEF2F1"],
    warm: ["#ff7c43", "#ffa600", "#FF6B6B", "#FFE66D", "#F7FFF7"],
    pastel: ["#a8e6cf", "#dcedc1", "#ffd3b6", "#ffaaa5", "#ff8b94"],
    dark: ["#1A535C", "#4ECDC4", "#FF6B6B", "#FFE66D", "#F7FFF7"],
  }

  // Annotation colors
  const annotationColors = [
    "#FF6B6B", // Red
    "#4ECDC4", // Teal
    "#FFE66D", // Yellow
    "#1A535C", // Dark Teal
    "#7B68EE", // Medium Slate Blue
    "#FF8C00", // Dark Orange
    "#32CD32", // Lime Green
    "#9370DB", // Medium Purple
  ]

  // Prepare data for visualization based on chart type and selected axes
  const prepareData = () => {
    if (!data || data.length === 0 || !xAxis || !yAxis) return []

    // For pie charts, we need to aggregate the data
    if (chartType === "pie") {
      const aggregatedData: Record<string, number> = {}

      data.forEach((item) => {
        const key = String(item[xAxis])
        if (!aggregatedData[key]) {
          aggregatedData[key] = 0
        }

        if (aggregationType === "count") {
          aggregatedData[key]++
        } else if (aggregationType === "sum") {
          aggregatedData[key] += Number(item[yAxis]) || 0
        } else if (aggregationType === "average") {
          if (!aggregatedData[`${key}_count`]) {
            aggregatedData[`${key}_count`] = 0
            aggregatedData[`${key}_sum`] = 0
          }
          aggregatedData[`${key}_count`]++
          aggregatedData[`${key}_sum`] += Number(item[yAxis]) || 0
        } else {
          // No aggregation, just use the first value
          if (aggregatedData[key] === 0) {
            aggregatedData[key] = Number(item[yAxis]) || 0
          }
        }
      })

      // Calculate averages if needed
      if (aggregationType === "average") {
        Object.keys(aggregatedData).forEach((key) => {
          if (key.endsWith("_count")) {
            const baseKey = key.replace("_count", "")
            if (aggregatedData[`${baseKey}_count`] > 0) {
              aggregatedData[baseKey] = aggregatedData[`${baseKey}_sum`] / aggregatedData[`${baseKey}_count`]
            }
            delete aggregatedData[`${baseKey}_count`]
            delete aggregatedData[`${baseKey}_sum`]
          }
        })
      }

      return Object.entries(aggregatedData).map(([name, value]) => ({ name, value }))
    }

    // For histograms (bar chart with binning)
    if (chartType === "bar" && columnTypes[xAxis] === "numeric" && aggregationType === "count") {
      // Find min and max values
      const values = data.map((item) => Number(item[xAxis])).filter((val) => !isNaN(val))
      const min = Math.min(...values)
      const max = Math.max(...values)

      // Create bins
      const binWidth = (max - min) / binCount
      const bins = Array(binCount)
        .fill(0)
        .map((_, i) => ({
          name: `${(min + i * binWidth).toFixed(1)} - ${(min + (i + 1) * binWidth).toFixed(1)}`,
          value: 0,
          binStart: min + i * binWidth,
          binEnd: min + (i + 1) * binWidth,
        }))

      // Count values in each bin
      values.forEach((val) => {
        const binIndex = Math.min(Math.floor((val - min) / binWidth), binCount - 1)
        if (binIndex >= 0) {
          bins[binIndex].value++
        }
      })

      return bins
    }

    // For other chart types, return the data as is with the selected axes
    return data.map((item) => ({
      x: item[xAxis],
      y: Number(item[yAxis]) || 0,
      ...(secondaryYAxis ? { y2: Number(item[secondaryYAxis]) || 0 } : {}),
      ...(colorBy ? { color: item[colorBy] } : {}),
      // Include the original item for tooltips
      original: item,
    }))
  }

  const visualizationData = useMemo(
    () => prepareData(),
    [data, xAxis, yAxis, secondaryYAxis, colorBy, chartType, aggregationType, binCount],
  )

  // Generate AI insights about the data
  const generateInsights = () => {
    setIsGeneratingInsights(true)

    // In a real implementation, this would call an AI service
    // For now, we'll simulate with some generic insights
    setTimeout(() => {
      const newInsights = []

      if (chartType === "bar" || chartType === "line") {
        if (yAxis && columnTypes[yAxis] === "numeric") {
          const values = data.map((item) => Number(item[yAxis])).filter((val) => !isNaN(val))
          const avg = values.reduce((sum, val) => sum + val, 0) / values.length
          const max = Math.max(...values)
          const min = Math.min(...values)

          newInsights.push(`The average ${yAxis} is ${avg.toFixed(2)}.`)
          newInsights.push(`The range of ${yAxis} is from ${min.toFixed(2)} to ${max.toFixed(2)}.`)

          if (chartType === "bar" && xAxis && columnTypes[xAxis] === "categorical") {
            // Find the category with the highest value
            const categoryValues: Record<string, number> = {}
            data.forEach((item) => {
              const category = String(item[xAxis])
              if (!categoryValues[category]) {
                categoryValues[category] = 0
              }
              categoryValues[category] += Number(item[yAxis]) || 0
            })

            const topCategory = Object.entries(categoryValues).sort((a, b) => b[1] - a[1])[0]

            if (topCategory) {
              newInsights.push(`"${topCategory[0]}" has the highest ${yAxis} at ${topCategory[1].toFixed(2)}.`)
            }
          }
        }
      } else if (chartType === "scatter") {
        if (yAxis && xAxis && columnTypes[yAxis] === "numeric" && columnTypes[xAxis] === "numeric") {
          // Calculate correlation
          const xValues = data.map((item) => Number(item[xAxis])).filter((val) => !isNaN(val))
          const yValues = data.map((item) => Number(item[yAxis])).filter((val) => !isNaN(val))

          if (xValues.length === yValues.length && xValues.length > 0) {
            const correlation = calculateCorrelation(xValues, yValues)

            if (correlation > 0.7) {
              newInsights.push(
                `There is a strong positive correlation (${correlation.toFixed(2)}) between ${xAxis} and ${yAxis}.`,
              )
            } else if (correlation < -0.7) {
              newInsights.push(
                `There is a strong negative correlation (${correlation.toFixed(2)}) between ${xAxis} and ${yAxis}.`,
              )
            } else if (correlation > 0.3) {
              newInsights.push(
                `There is a moderate positive correlation (${correlation.toFixed(2)}) between ${xAxis} and ${yAxis}.`,
              )
            } else if (correlation < -0.3) {
              newInsights.push(
                `There is a moderate negative correlation (${correlation.toFixed(2)}) between ${xAxis} and ${yAxis}.`,
              )
            } else {
              newInsights.push(`There is a weak correlation (${correlation.toFixed(2)}) between ${xAxis} and ${yAxis}.`)
            }
          }
        }
      } else if (chartType === "pie") {
        if (xAxis && columnTypes[xAxis] === "categorical") {
          // Count occurrences of each category
          const categoryCounts: Record<string, number> = {}
          data.forEach((item) => {
            const category = String(item[xAxis])
            if (!categoryCounts[category]) {
              categoryCounts[category] = 0
            }
            categoryCounts[category]++
          })

          const totalCount = Object.values(categoryCounts).reduce((sum, count) => sum + count, 0)
          const topCategory = Object.entries(categoryCounts).sort((a, b) => b[1] - a[1])[0]

          if (topCategory) {
            const percentage = ((topCategory[1] / totalCount) * 100).toFixed(1)
            newInsights.push(`"${topCategory[0]}" is the most common ${xAxis} at ${percentage}% of the data.`)
          }
        }
      }

      // Add some general insights
      newInsights.push(`The dataset contains ${data.length} records.`)

      const numericColumns = columns.filter((col) => columnTypes[col] === "numeric")
      if (numericColumns.length > 0) {
        newInsights.push(`There are ${numericColumns.length} numeric columns that can be used for analysis.`)
      }

      // Suggest annotations for key points
      if (chartType === "line" || chartType === "bar" || chartType === "area") {
        if (yAxis && columnTypes[yAxis] === "numeric") {
          const values = data.map((item) => Number(item[yAxis])).filter((val) => !isNaN(val))
          const max = Math.max(...values)
          const maxItem = data.find((item) => Number(item[yAxis]) === max)

          if (maxItem) {
            newInsights.push(`Consider adding an annotation at ${maxItem[xAxis]} to highlight the peak ${yAxis} value.`)
          }
        }
      }

      setInsights(newInsights)
      setIsGeneratingInsights(false)
    }, 1500)
  }

  // Calculate Pearson correlation coefficient
  const calculateCorrelation = (x: number[], y: number[]) => {
    const n = x.length
    let sumX = 0,
      sumY = 0,
      sumXY = 0,
      sumX2 = 0,
      sumY2 = 0

    for (let i = 0; i < n; i++) {
      sumX += x[i]
      sumY += y[i]
      sumXY += x[i] * y[i]
      sumX2 += x[i] * x[i]
      sumY2 += y[i] * y[i]
    }

    const numerator = n * sumXY - sumX * sumY
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))

    return denominator === 0 ? 0 : numerator / denominator
  }

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: TooltipProps<any, any>) => {
    if (active && payload && payload.length) {
      if (chartType === "pie") {
        const data = payload[0].payload
        return (
          <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
            <p className="font-medium">{data.name}</p>
            <p className="text-sm">
              <span className="font-medium">Value:</span> {data.value.toLocaleString()}
            </p>
            <p className="text-sm">
              <span className="font-medium">Percentage:</span> {(data.percent * 100).toFixed(1)}%
            </p>
            {isAddingAnnotation && (
              <Button size="sm" variant="outline" className="mt-2" onClick={() => handleSelectDataPoint(data)}>
                <MapPin className="h-3 w-3 mr-1" /> Annotate
              </Button>
            )}
          </div>
        )
      }

      return (
        <div className="bg-background border rounded-lg shadow-lg p-3 text-sm">
          <p className="font-medium">{label || payload[0].payload.x}</p>
          {payload.map((entry, index) => {
            const dataKey = entry.dataKey === "y" ? yAxis : entry.dataKey === "y2" ? secondaryYAxis : entry.name
            return (
              <p key={`item-${index}`} className="text-sm">
                <span
                  className="inline-block w-3 h-3 rounded-full mr-2"
                  style={{ backgroundColor: entry.color }}
                ></span>
                <span className="font-medium">{dataKey}:</span> {entry.value.toLocaleString()}
              </p>
            )
          })}
          {payload[0].payload.original && colorBy && (
            <p className="text-sm">
              <span className="font-medium">{colorBy}:</span> {payload[0].payload.original[colorBy]}
            </p>
          )}
          {isAddingAnnotation && (
            <Button
              size="sm"
              variant="outline"
              className="mt-2"
              onClick={() => handleSelectDataPoint(payload[0].payload)}
            >
              <MapPin className="h-3 w-3 mr-1" /> Annotate
            </Button>
          )}
        </div>
      )
    }

    return null
  }

  // Handle selecting a data point for annotation
  const handleSelectDataPoint = (dataPoint: any) => {
    setSelectedDataPoint(dataPoint)

    // Pre-fill the new annotation form
    const xVal = dataPoint.x || dataPoint.name
    const yVal = dataPoint.y || dataPoint.value

    setNewAnnotation({
      ...newAnnotation,
      xValue: xVal,
      yValue: yVal,
      label: `${xVal}: ${yVal}`,
    })
  }

  // Add a new annotation
  const handleAddAnnotation = () => {
    if (!newAnnotation.xValue || !newAnnotation.label) return

    const annotation: Annotation = {
      id: Date.now().toString(),
      xValue: newAnnotation.xValue,
      yValue: newAnnotation.yValue,
      label: newAnnotation.label || "",
      color: newAnnotation.color || "#FF6B6B",
      position: newAnnotation.position || "top",
      important: newAnnotation.important || false,
    }

    setAnnotations([...annotations, annotation])
    setNewAnnotation({
      color: "#FF6B6B",
      position: "top",
      important: false,
    })
    setSelectedDataPoint(null)
    setIsAddingAnnotation(false)
  }

  // Update an existing annotation
  const handleUpdateAnnotation = () => {
    if (!editingAnnotation) return

    const updatedAnnotations = annotations.map((ann) => (ann.id === editingAnnotation.id ? editingAnnotation : ann))

    setAnnotations(updatedAnnotations)
    setEditingAnnotation(null)
  }

  // Delete an annotation
  const handleDeleteAnnotation = (id: string) => {
    setAnnotations(annotations.filter((ann) => ann.id !== id))
  }

  // Start editing an annotation
  const handleEditAnnotation = (annotation: Annotation) => {
    setEditingAnnotation(annotation)
  }

  // Cancel editing
  const handleCancelEdit = () => {
    setEditingAnnotation(null)
  }

  // Render the appropriate chart based on the selected chart type
  const renderChart = () => {
    const colors = colorSchemes[colorScheme] || colorSchemes.neon

    if (!xAxis || !yAxis || visualizationData.length === 0) {
      return (
        <div className="flex items-center justify-center h-[400px] text-muted-foreground">
          <div className="text-center">
            <p>Select axes to visualize data</p>
          </div>
        </div>
      )
    }

    // Filter annotations to only show those that match the current axes
    const filteredAnnotations = showAnnotations
      ? annotations.filter((ann) => {
          // For pie charts, match on name
          if (chartType === "pie") {
            return visualizationData.some((d) => d.name === ann.xValue)
          }
          // For other charts, match on x and y
          return visualizationData.some((d) => d.x === ann.xValue)
        })
      : []

    switch (chartType) {
      case "bar":
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsBarChart
                data={visualizationData}
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                onClick={(data) => {
                  if (isAddingAnnotation && data && data.activePayload) {
                    handleSelectDataPoint(data.activePayload[0].payload)
                  }
                }}
              >
                {showGrid && <CartesianGrid strokeDasharray="3 3" />}
                <XAxis
                  dataKey={
                    chartType === "bar" && columnTypes[xAxis] === "numeric" && aggregationType === "count"
                      ? "name"
                      : "x"
                  }
                  angle={-45}
                  textAnchor="end"
                  height={70}
                />
                <YAxis />
                <RechartsTooltip content={<CustomTooltip />} />
                {showLegend && <Legend />}
                <Bar
                  dataKey={
                    chartType === "bar" && columnTypes[xAxis] === "numeric" && aggregationType === "count"
                      ? "value"
                      : "y"
                  }
                  name={yAxis}
                  fill={colors[0]}
                />
                {secondaryYAxis && <Bar dataKey="y2" name={secondaryYAxis} fill={colors[1]} />}

                {/* Render annotations */}
                {filteredAnnotations.map((ann) => (
                  <ReferenceDot
                    key={ann.id}
                    x={ann.xValue}
                    y={ann.yValue}
                    r={6}
                    fill={ann.color}
                    stroke="white"
                    strokeWidth={2}
                  >
                    <ReferenceLabel
                      value={ann.label}
                      position={ann.position}
                      fill={ann.color}
                      fontSize={12}
                      fontWeight={ann.important ? "bold" : "normal"}
                      className="annotation-label"
                    />
                  </ReferenceDot>
                ))}
              </RechartsBarChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      case "line":
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsLineChart
                data={visualizationData}
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                onClick={(data) => {
                  if (isAddingAnnotation && data && data.activePayload) {
                    handleSelectDataPoint(data.activePayload[0].payload)
                  }
                }}
              >
                {showGrid && <CartesianGrid strokeDasharray="3 3" />}
                <XAxis dataKey="x" angle={-45} textAnchor="end" height={70} />
                <YAxis />
                <RechartsTooltip content={<CustomTooltip />} />
                {showLegend && <Legend />}
                <Line type="monotone" dataKey="y" name={yAxis} stroke={colors[0]} activeDot={{ r: 8 }} />
                {secondaryYAxis && <Line type="monotone" dataKey="y2" name={secondaryYAxis} stroke={colors[1]} />}

                {/* Render annotations */}
                {filteredAnnotations.map((ann) => (
                  <ReferenceDot
                    key={ann.id}
                    x={ann.xValue}
                    y={ann.yValue}
                    r={6}
                    fill={ann.color}
                    stroke="white"
                    strokeWidth={2}
                  >
                    <ReferenceLabel
                      value={ann.label}
                      position={ann.position}
                      fill={ann.color}
                      fontSize={12}
                      fontWeight={ann.important ? "bold" : "normal"}
                      className="annotation-label"
                    />
                  </ReferenceDot>
                ))}
              </RechartsLineChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      case "scatter":
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                onClick={(data) => {
                  if (isAddingAnnotation && data && data.activePayload) {
                    handleSelectDataPoint(data.activePayload[0].payload)
                  }
                }}
              >
                {showGrid && <CartesianGrid strokeDasharray="3 3" />}
                <XAxis type="number" dataKey="x" name={xAxis} angle={-45} textAnchor="end" height={70} />
                <YAxis type="number" dataKey="y" name={yAxis} />
                <ZAxis range={[60, 400]} />
                <RechartsTooltip content={<CustomTooltip />} />
                {showLegend && <Legend />}
                <Scatter name={yAxis} data={visualizationData} fill={colors[0]} />

                {/* Render annotations */}
                {filteredAnnotations.map((ann) => (
                  <ReferenceDot
                    key={ann.id}
                    x={ann.xValue}
                    y={ann.yValue}
                    r={6}
                    fill={ann.color}
                    stroke="white"
                    strokeWidth={2}
                  >
                    <ReferenceLabel
                      value={ann.label}
                      position={ann.position}
                      fill={ann.color}
                      fontSize={12}
                      fontWeight={ann.important ? "bold" : "normal"}
                      className="annotation-label"
                    />
                  </ReferenceDot>
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      case "pie":
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                onClick={(data) => {
                  if (isAddingAnnotation && data && data.activePayload) {
                    handleSelectDataPoint(data.activePayload[0].payload)
                  }
                }}
              >
                <Pie
                  data={visualizationData}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  outerRadius={150}
                  fill="#8884d8"
                  dataKey="value"
                  nameKey="name"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {visualizationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Pie>
                <RechartsTooltip content={<CustomTooltip />} />
                {showLegend && <Legend />}

                {/* For pie charts, annotations are rendered differently */}
                {filteredAnnotations.map((ann) => {
                  // Find the corresponding data point
                  const dataPoint = visualizationData.find((d) => d.name === ann.xValue)
                  if (!dataPoint) return null

                  // Calculate position (simplified)
                  const angle = Math.PI / 4 // Arbitrary angle for demonstration
                  const radius = 170 // Slightly larger than the pie
                  const x = Math.cos(angle) * radius + 200 // Assuming center at (200, 200)
                  const y = Math.sin(angle) * radius + 200

                  return (
                    <foreignObject key={ann.id} x={x - 50} y={y - 25} width={100} height={50}>
                      <div
                        style={{
                          backgroundColor: ann.color,
                          color: "white",
                          padding: "4px 8px",
                          borderRadius: "4px",
                          fontSize: "12px",
                          fontWeight: ann.important ? "bold" : "normal",
                          textAlign: "center",
                        }}
                      >
                        {ann.label}
                      </div>
                    </foreignObject>
                  )
                })}
              </RechartsPieChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      case "area":
        return (
          <ChartContainer className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsAreaChart
                data={visualizationData}
                margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                onClick={(data) => {
                  if (isAddingAnnotation && data && data.activePayload) {
                    handleSelectDataPoint(data.activePayload[0].payload)
                  }
                }}
              >
                {showGrid && <CartesianGrid strokeDasharray="3 3" />}
                <XAxis dataKey="x" angle={-45} textAnchor="end" height={70} />
                <YAxis />
                <RechartsTooltip content={<CustomTooltip />} />
                {showLegend && <Legend />}
                <Area type="monotone" dataKey="y" name={yAxis} stroke={colors[0]} fill={colors[0]} fillOpacity={0.3} />
                {secondaryYAxis && (
                  <Area
                    type="monotone"
                    dataKey="y2"
                    name={secondaryYAxis}
                    stroke={colors[1]}
                    fill={colors[1]}
                    fillOpacity={0.3}
                  />
                )}

                {/* Render annotations */}
                {filteredAnnotations.map((ann) => (
                  <ReferenceDot
                    key={ann.id}
                    x={ann.xValue}
                    y={ann.yValue}
                    r={6}
                    fill={ann.color}
                    stroke="white"
                    strokeWidth={2}
                  >
                    <ReferenceLabel
                      value={ann.label}
                      position={ann.position}
                      fill={ann.color}
                      fontSize={12}
                      fontWeight={ann.important ? "bold" : "normal"}
                      className="annotation-label"
                    />
                  </ReferenceDot>
                ))}
              </RechartsAreaChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      default:
        return (
          <div className="flex items-center justify-center h-[400px] text-muted-foreground">
            <div className="text-center">
              <p>Select a chart type to visualize data</p>
            </div>
          </div>
        )
    }
  }

  // Render annotation form
  const renderAnnotationForm = () => {
    return (
      <div className="space-y-4">
        {selectedDataPoint ? (
          <div className="space-y-4 border p-4 rounded-md bg-muted/30">
            <div className="flex justify-between items-center">
              <h4 className="text-sm font-medium">New Annotation</h4>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setSelectedDataPoint(null)
                  setNewAnnotation({
                    color: "#FF6B6B",
                    position: "top",
                    important: false,
                  })
                }}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="space-y-2">
              <Label htmlFor="annotation-label">Label</Label>
              <Textarea
                id="annotation-label"
                value={newAnnotation.label || ""}
                onChange={(e) => setNewAnnotation({ ...newAnnotation, label: e.target.value })}
                placeholder="Enter annotation text"
                className="resize-none"
              />
            </div>

            <div className="space-y-2">
              <Label>Position</Label>
              <div className="grid grid-cols-4 gap-2">
                {(["top", "right", "bottom", "left"] as const).map((pos) => (
                  <Button
                    key={pos}
                    type="button"
                    variant={newAnnotation.position === pos ? "default" : "outline"}
                    size="sm"
                    onClick={() => setNewAnnotation({ ...newAnnotation, position: pos })}
                  >
                    {pos.charAt(0).toUpperCase() + pos.slice(1)}
                  </Button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Color</Label>
              <div className="grid grid-cols-8 gap-2">
                {annotationColors.map((color) => (
                  <Button
                    key={color}
                    type="button"
                    variant="outline"
                    size="sm"
                    className={`h-8 p-0 ${newAnnotation.color === color ? "ring-2 ring-ring" : ""}`}
                    style={{ backgroundColor: color }}
                    onClick={() => setNewAnnotation({ ...newAnnotation, color })}
                  >
                    <span className="sr-only">Select {color}</span>
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="important"
                checked={newAnnotation.important || false}
                onCheckedChange={(checked) => setNewAnnotation({ ...newAnnotation, important: checked })}
              />
              <Label htmlFor="important">Mark as important</Label>
            </div>

            <div className="flex justify-end space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setSelectedDataPoint(null)
                  setNewAnnotation({
                    color: "#FF6B6B",
                    position: "top",
                    important: false,
                  })
                }}
              >
                Cancel
              </Button>
              <Button size="sm" onClick={handleAddAnnotation} disabled={!newAnnotation.label}>
                Add Annotation
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <Switch id="show-annotations" checked={showAnnotations} onCheckedChange={setShowAnnotations} />
              <Label htmlFor="show-annotations">Show annotations</Label>
            </div>

            <Button
              size="sm"
              onClick={() => setIsAddingAnnotation(!isAddingAnnotation)}
              variant={isAddingAnnotation ? "default" : "outline"}
            >
              {isAddingAnnotation ? (
                <>
                  <X className="h-4 w-4 mr-2" />
                  Cancel
                </>
              ) : (
                <>
                  <Plus className="h-4 w-4 mr-2" />
                  Add Annotation
                </>
              )}
            </Button>
          </div>
        )}

        {isAddingAnnotation && !selectedDataPoint && (
          <div className="p-3 bg-muted/30 rounded-md text-sm">
            Click on a data point in the chart to add an annotation.
          </div>
        )}

        {annotations.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Existing Annotations</h4>
            <ScrollArea className="h-[200px] rounded-md border">
              <div className="p-4 space-y-2">
                {annotations.map((ann) => (
                  <div
                    key={ann.id}
                    className={`p-3 rounded-md border ${editingAnnotation?.id === ann.id ? "border-primary" : ""}`}
                  >
                    {editingAnnotation?.id === ann.id ? (
                      <div className="space-y-3">
                        <Textarea
                          value={editingAnnotation.label}
                          onChange={(e) =>
                            setEditingAnnotation({
                              ...editingAnnotation,
                              label: e.target.value,
                            })
                          }
                          className="resize-none"
                        />

                        <div className="grid grid-cols-4 gap-2">
                          {(["top", "right", "bottom", "left"] as const).map((pos) => (
                            <Button
                              key={pos}
                              type="button"
                              variant={editingAnnotation.position === pos ? "default" : "outline"}
                              size="sm"
                              onClick={() =>
                                setEditingAnnotation({
                                  ...editingAnnotation,
                                  position: pos,
                                })
                              }
                            >
                              {pos.charAt(0).toUpperCase() + pos.slice(1)}
                            </Button>
                          ))}
                        </div>

                        <div className="grid grid-cols-8 gap-2">
                          {annotationColors.map((color) => (
                            <Button
                              key={color}
                              type="button"
                              variant="outline"
                              size="sm"
                              className={`h-8 p-0 ${editingAnnotation.color === color ? "ring-2 ring-ring" : ""}`}
                              style={{ backgroundColor: color }}
                              onClick={() =>
                                setEditingAnnotation({
                                  ...editingAnnotation,
                                  color,
                                })
                              }
                            >
                              <span className="sr-only">Select {color}</span>
                            </Button>
                          ))}
                        </div>

                        <div className="flex items-center space-x-2">
                          <Switch
                            id={`important-${ann.id}`}
                            checked={editingAnnotation.important}
                            onCheckedChange={(checked) =>
                              setEditingAnnotation({
                                ...editingAnnotation,
                                important: checked,
                              })
                            }
                          />
                          <Label htmlFor={`important-${ann.id}`}>Mark as important</Label>
                        </div>

                        <div className="flex justify-end space-x-2">
                          <Button variant="outline" size="sm" onClick={handleCancelEdit}>
                            Cancel
                          </Button>
                          <Button size="sm" onClick={handleUpdateAnnotation}>
                            Save
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-start justify-between">
                          <div className="flex items-center space-x-2" style={{ color: ann.color }}>
                            <MapPin className="h-4 w-4" />
                            <span className={`font-medium ${ann.important ? "font-bold" : ""}`}>{ann.xValue}</span>
                          </div>
                          <div className="flex space-x-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6"
                              onClick={() => handleEditAnnotation(ann)}
                            >
                              <Pencil className="h-3 w-3" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-destructive"
                              onClick={() => handleDeleteAnnotation(ann.id)}
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                        <p className="mt-1 text-sm">{ann.label}</p>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="chart" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="chart">Chart Builder</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="annotations">Annotations</TabsTrigger>
          <TabsTrigger value="settings">Chart Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="chart" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-4">
                <h3 className="text-sm font-medium mb-3">Chart Type</h3>
                <div className="grid grid-cols-3 gap-2">
                  <Button
                    variant={chartType === "bar" ? "default" : "outline"}
                    size="sm"
                    className="flex flex-col h-auto py-2 px-1"
                    onClick={() => setChartType("bar")}
                  >
                    <BarChart className="h-4 w-4 mb-1" />
                    <span className="text-xs">Bar</span>
                  </Button>
                  <Button
                    variant={chartType === "line" ? "default" : "outline"}
                    size="sm"
                    className="flex flex-col h-auto py-2 px-1"
                    onClick={() => setChartType("line")}
                  >
                    <LineChart className="h-4 w-4 mb-1" />
                    <span className="text-xs">Line</span>
                  </Button>
                  <Button
                    variant={chartType === "scatter" ? "default" : "outline"}
                    size="sm"
                    className="flex flex-col h-auto py-2 px-1"
                    onClick={() => setChartType("scatter")}
                  >
                    <ScatterChartIcon className="h-4 w-4 mb-1" />
                    <span className="text-xs">Scatter</span>
                  </Button>
                  <Button
                    variant={chartType === "pie" ? "default" : "outline"}
                    size="sm"
                    className="flex flex-col h-auto py-2 px-1"
                    onClick={() => setChartType("pie")}
                  >
                    <PieChart className="h-4 w-4 mb-1" />
                    <span className="text-xs">Pie</span>
                  </Button>
                  <Button
                    variant={chartType === "area" ? "default" : "outline"}
                    size="sm"
                    className="flex flex-col h-auto py-2 px-1"
                    onClick={() => setChartType("area")}
                  >
                    <AreaChart className="h-4 w-4 mb-1" />
                    <span className="text-xs">Area</span>
                  </Button>
                </div>

                <div className="space-y-3 mt-4">
                  <div className="space-y-1">
                    <Label htmlFor="x-axis">X-Axis</Label>
                    <Select value={xAxis} onValueChange={setXAxis}>
                      <SelectTrigger id="x-axis">
                        <SelectValue placeholder="Select X-Axis" />
                      </SelectTrigger>
                      <SelectContent>
                        {columns.map((column) => (
                          <SelectItem key={column} value={column}>
                            {column} {columnTypes[column] ? `(${columnTypes[column]})` : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1">
                    <Label htmlFor="y-axis">Y-Axis</Label>
                    <Select value={yAxis} onValueChange={setYAxis}>
                      <SelectTrigger id="y-axis">
                        <SelectValue placeholder="Select Y-Axis" />
                      </SelectTrigger>
                      <SelectContent>
                        {columns.map((column) => (
                          <SelectItem key={column} value={column}>
                            {column} {columnTypes[column] ? `(${columnTypes[column]})` : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {(chartType === "line" || chartType === "bar" || chartType === "area") && (
                    <div className="space-y-1">
                      <Label htmlFor="secondary-y-axis">Secondary Y-Axis (Optional)</Label>
                      <Select value={secondaryYAxis} onValueChange={setSecondaryYAxis}>
                        <SelectTrigger id="secondary-y-axis">
                          <SelectValue placeholder="Select Secondary Y-Axis" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          {columns
                            .filter((column) => column !== yAxis && columnTypes[column] === "numeric")
                            .map((column) => (
                              <SelectItem key={column} value={column}>
                                {column} (numeric)
                              </SelectItem>
                            ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {chartType === "scatter" && (
                    <div className="space-y-1">
                      <Label htmlFor="color-by">Color By (Optional)</Label>
                      <Select value={colorBy} onValueChange={setColorBy}>
                        <SelectTrigger id="color-by">
                          <SelectValue placeholder="Select Color Field" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          {columns
                            .filter((column) => column !== xAxis && column !== yAxis)
                            .map((column) => (
                              <SelectItem key={column} value={column}>
                                {column} {columnTypes[column] ? `(${columnTypes[column]})` : ""}
                              </SelectItem>
                            ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {(chartType === "bar" && columnTypes[xAxis] === "numeric") || chartType === "pie" ? (
                    <div className="space-y-1">
                      <Label htmlFor="aggregation-type">Aggregation</Label>
                      <Select value={aggregationType} onValueChange={setAggregationType as any}>
                        <SelectTrigger id="aggregation-type">
                          <SelectValue placeholder="Select Aggregation" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None</SelectItem>
                          <SelectItem value="sum">Sum</SelectItem>
                          <SelectItem value="average">Average</SelectItem>
                          <SelectItem value="count">Count</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  ) : null}

                  {chartType === "bar" && columnTypes[xAxis] === "numeric" && aggregationType === "count" && (
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <Label htmlFor="bin-count">Number of Bins</Label>
                        <span className="text-sm">{binCount}</span>
                      </div>
                      <Slider
                        id="bin-count"
                        min={5}
                        max={30}
                        step={1}
                        value={[binCount]}
                        onValueChange={(value) => setBinCount(value[0])}
                      />
                    </div>
                  )}
                </div>

                <div className="mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full"
                    onClick={generateInsights}
                    disabled={isGeneratingInsights || !xAxis || !yAxis}
                  >
                    {isGeneratingInsights ? (
                      <>
                        <RefreshCw className="h-3 w-3 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Wand2 className="h-3 w-3 mr-2" />
                        Generate Insights
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            <div className="md:col-span-2">
              <Card className="h-full">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-medium">{chartTitle}</h3>
                    <div className="flex items-center space-x-2">
                      {annotations.length > 0 && (
                        <div className="flex items-center space-x-2 mr-2">
                          <Switch
                            id="toggle-annotations"
                            checked={showAnnotations}
                            onCheckedChange={setShowAnnotations}
                            className="data-[state=checked]:bg-primary"
                          />
                          <Label htmlFor="toggle-annotations" className="text-xs">
                            Annotations
                          </Label>
                        </div>
                      )}
                      <TooltipProvider>
                        <UITooltip>
                          <TooltipTrigger asChild>
                            <Button variant="outline" size="icon">
                              <Download className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Download Chart</p>
                          </TooltipContent>
                        </UITooltip>
                      </TooltipProvider>

                      <TooltipProvider>
                        <UITooltip>
                          <TooltipTrigger asChild>
                            <Button variant="outline" size="icon">
                              <Save className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Save Chart</p>
                          </TooltipContent>
                        </UITooltip>
                      </TooltipProvider>
                    </div>
                  </div>

                  <div ref={chartRef}>{renderChart()}</div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="insights" className="space-y-4 mt-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium">AI-Generated Insights</h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={generateInsights}
                  disabled={isGeneratingInsights || !xAxis || !yAxis}
                >
                  {isGeneratingInsights ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Wand2 className="h-4 w-4 mr-2" />
                      Refresh Insights
                    </>
                  )}
                </Button>
              </div>

              {isGeneratingInsights ? (
                <div className="flex items-center justify-center h-40">
                  <div className="flex flex-col items-center">
                    <RefreshCw className="h-8 w-8 animate-spin text-primary mb-4" />
                    <p>Analyzing your data...</p>
                  </div>
                </div>
              ) : insights.length > 0 ? (
                <div className="space-y-3">
                  {insights.map((insight, index) => (
                    <div key={index} className="flex items-start p-3 rounded-md bg-muted/30">
                      <Wand2 className="h-5 w-5 text-primary mr-3 mt-0.5" />
                      <p>{insight}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center h-40 text-muted-foreground">
                  <div className="text-center">
                    <p>Click "Generate Insights" to analyze your data</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h3 className="text-lg font-medium mb-4">Data Summary</h3>

              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium mb-2">Dataset Overview</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="bg-muted/30 p-3 rounded-md">
                      <p className="text-sm text-muted-foreground">Total Records</p>
                      <p className="text-2xl font-bold">{data.length}</p>
                    </div>
                    <div className="bg-muted/30 p-3 rounded-md">
                      <p className="text-sm text-muted-foreground">Columns</p>
                      <p className="text-2xl font-bold">{columns.length}</p>
                    </div>
                    <div className="bg-muted/30 p-3 rounded-md">
                      <p className="text-sm text-muted-foreground">Numeric Columns</p>
                      <p className="text-2xl font-bold">
                        {columns.filter((col) => columnTypes[col] === "numeric").length}
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium mb-2">Column Types</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {columns.map((column) => (
                      <div key={column} className="flex items-center justify-between p-2 border rounded-md">
                        <span className="font-medium">{column}</span>
                        <Badge variant="outline">{columnTypes[column] || "unknown"}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="annotations" className="space-y-4 mt-4">
          <Card>
            <CardContent className="p-6">
              <h3 className="text-lg font-medium mb-4">Chart Annotations</h3>
              {renderAnnotationForm()}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4 mt-4">
          <Card>
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-medium mb-2">Chart Appearance</h3>

                  <div className="space-y-2">
                    <Label htmlFor="chart-title">Chart Title</Label>
                    <Input id="chart-title" value={chartTitle} onChange={(e) => setChartTitle(e.target.value)} />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="color-scheme">Color Scheme</Label>
                    <Select value={colorScheme} onValueChange={setColorScheme}>
                      <SelectTrigger id="color-scheme">
                        <SelectValue placeholder="Select Color Scheme" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="neon">Neon</SelectItem>
                        <SelectItem value="blues">Blues</SelectItem>
                        <SelectItem value="greens">Greens</SelectItem>
                        <SelectItem value="warm">Warm</SelectItem>
                        <SelectItem value="pastel">Pastel</SelectItem>
                        <SelectItem value="dark">Dark</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center justify-between">
                    <Label htmlFor="show-grid">Show Grid</Label>
                    <Switch id="show-grid" checked={showGrid} onCheckedChange={setShowGrid} />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label htmlFor="show-legend">Show Legend</Label>
                    <Switch id="show-legend" checked={showLegend} onCheckedChange={setShowLegend} />
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="text-lg font-medium mb-2">Color Preview</h3>

                  <div className="grid grid-cols-5 gap-2">
                    {(colorSchemes[colorScheme] || colorSchemes.neon).map((color, index) => (
                      <div
                        key={index}
                        className="h-8 rounded-md flex items-center justify-center text-xs font-medium text-white"
                        style={{ backgroundColor: color }}
                      >
                        {index + 1}
                      </div>
                    ))}
                  </div>

                  <div className="mt-6">
                    <h3 className="text-lg font-medium mb-2">Export Options</h3>

                    <div className="grid grid-cols-2 gap-2">
                      <Button variant="outline" className="w-full">
                        <Download className="mr-2 h-4 w-4" />
                        PNG
                      </Button>
                      <Button variant="outline" className="w-full">
                        <Download className="mr-2 h-4 w-4" />
                        SVG
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
