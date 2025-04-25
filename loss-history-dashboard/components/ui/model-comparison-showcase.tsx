"use client"

import React from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { motion } from "framer-motion"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts'
import { ArrowRight, TrendingUp, Zap, BarChart3, PieChart } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

// Sample data for the comparison
const modelComparisonData = [
  {
    name: 'R² Score',
    Model1: 0.68,
    Model4: 0.802,
  },
  {
    name: 'RMSE',
    Model1: 0.42,
    Model4: 0.31,
  },
  {
    name: 'MAE',
    Model1: 0.38,
    Model4: 0.29,
  },
]

const businessImpactData = [
  {
    title: "Pricing Precision",
    description: "More accurate risk assessment leads to better policy pricing.",
    improvement: "+22%",
    icon: <TrendingUp className="h-6 w-6 text-blue-500" />,
  },
  {
    title: "Loss Prediction",
    description: "Enhanced ability to predict potential losses.",
    improvement: "+17.9%",
    icon: <Zap className="h-6 w-6 text-amber-500" />,
  },
  {
    title: "Competitive Edge",
    description: "Advantage in market positioning through better risk assessment.",
    improvement: "+15%",
    icon: <BarChart3 className="h-6 w-6 text-green-500" />,
  },
  {
    title: "Customer Retention",
    description: "Increased customer satisfaction through fair pricing.",
    improvement: "+12%",
    icon: <PieChart className="h-6 w-6 text-purple-500" />,
  },
]

const timeSeriesData = Array.from({ length: 12 }, (_, i) => ({
  month: `Month ${i + 1}`,
  model1Prediction: 100 + Math.random() * 50,
  model4Prediction: 110 + Math.random() * 40,
  actualLoss: 105 + Math.random() * 45,
}))

export function ModelComparisonShowcase() {
  return (
    <section className="w-full py-12 md:py-16 lg:py-20 bg-gray-50 dark:bg-gray-900">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center justify-center space-y-4 text-center mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl text-primary">
              Model Performance Comparison
            </h2>
            <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400 mt-4">
              See how our enhanced Model4 with fast decay weighting outperforms traditional Model1 for homeowner loss predictions.
            </p>
          </motion.div>
        </div>

        {/* R² Score Comparison Visualization */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-12"
        >
          <Card className="shadow-lg border-gray-200 dark:border-gray-800">
            <CardHeader>
              <CardTitle className="text-2xl">Performance Metrics Comparison</CardTitle>
              <CardDescription>
                Model4 shows significant improvement across key performance indicators
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={modelComparisonData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [value.toFixed(3), ""]}
                      labelFormatter={(label) => `Metric: ${label}`}
                    />
                    <Legend />
                    <Bar name="Model1 (Traditional)" dataKey="Model1" fill="#8884d8" />
                    <Bar name="Model4 (Enhanced)" dataKey="Model4" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  <strong>Model4 shows a 17.9% improvement in R² score</strong> over the traditional model, resulting in more precise loss predictions.
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Business Impact Analysis */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mb-12"
        >
          <h3 className="text-2xl font-bold text-center mb-8">Business Impact</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {businessImpactData.map((item, i) => (
              <Card key={i} className={cn("shadow-md transition-all hover:shadow-lg", 
                i === 0 ? "border-blue-200 dark:border-blue-800" : 
                i === 1 ? "border-amber-200 dark:border-amber-800" : 
                i === 2 ? "border-green-200 dark:border-green-800" :
                "border-purple-200 dark:border-purple-800"
              )}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    {item.icon}
                    <span className="text-xl font-bold text-green-600 dark:text-green-400">{item.improvement}</span>
                  </div>
                  <CardTitle className="text-lg">{item.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{item.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </motion.div>

        {/* Before/After Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <Card className="shadow-lg border-gray-200 dark:border-gray-800">
            <CardHeader>
              <CardTitle className="text-2xl">Prediction Accuracy Over Time</CardTitle>
              <CardDescription>
                See how Model4 closely follows actual loss patterns compared to Model1
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={timeSeriesData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="actualLoss" 
                      stroke="#ff7300" 
                      name="Actual Loss"
                      strokeWidth={2}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="model1Prediction" 
                      stroke="#8884d8" 
                      name="Model1 Prediction"
                      strokeDasharray="5 5"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="model4Prediction" 
                      stroke="#82ca9d" 
                      name="Model4 Prediction"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
            <CardFooter className="flex justify-center">
              <Button className="mt-4" variant="outline">
                View Detailed Analysis <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    </section>
  )
}

