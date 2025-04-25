"use client"

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  Area,
  ComposedChart,
  AreaChart
} from 'recharts'

// Generate projection data from 2024 to 2034
const generateMonthlyData = () => {
  const data = []
  const baseCustomers = 15000
  const baseSales = 2500000
  const baseProfit = 850000
  const baseRevenue = 3200000

  // Starting from January 2024
  for (let year = 2024; year <= 2034; year++) {
    for (let month = 0; month < 12; month++) {
      // Skip future months in 2034
      if (year === 2034 && month > new Date().getMonth()) break

      // Growth factors (compounding growth with seasonal variations)
      const monthsSince2024 = (year - 2024) * 12 + month
      const yearFactor = 1 + (year - 2024) * 0.05 // 5% yearly base growth
      const seasonalFactor = 1 + Math.sin(month / 11 * Math.PI) * 0.15 // Seasonal variation
      const randomFactor = 0.97 + Math.random() * 0.06 // Random variation ±3%
      
      // Add occasional spikes for marketing campaigns
      const campaignSpike = (month === 3 || month === 9) && Math.random() > 0.7 ? 1.15 : 1

      const customers = Math.round(baseCustomers * yearFactor * (1 + monthsSince2024 * 0.003) * seasonalFactor * randomFactor * campaignSpike)
      const sales = Math.round(baseSales * yearFactor * (1 + monthsSince2024 * 0.004) * seasonalFactor * randomFactor * campaignSpike)
      const profit = Math.round(baseProfit * yearFactor * (1 + monthsSince2024 * 0.0035) * seasonalFactor * randomFactor * campaignSpike)
      const revenue = Math.round(baseRevenue * yearFactor * (1 + monthsSince2024 * 0.0045) * seasonalFactor * randomFactor * campaignSpike)
      
      data.push({
        date: `${year}-${String(month + 1).padStart(2, '0')}`,
        customers,
        sales,
        profit,
        revenue,
        roi: Math.round((profit / sales) * 100 * 10) / 10, // ROI as percentage with 1 decimal
        monthLabel: new Date(year, month).toLocaleString('default', { month: 'short' }),
        year,
        month: month + 1
      })
    }
  }
  
  return data
}

// Generate yearly data by aggregating monthly data
const generateYearlyData = (monthlyData) => {
  const yearlyData = []
  const yearGroups = {}
  
  monthlyData.forEach(item => {
    if (!yearGroups[item.year]) {
      yearGroups[item.year] = {
        year: item.year,
        customers: 0,
        sales: 0,
        profit: 0,
        revenue: 0,
        months: 0
      }
    }
    
    yearGroups[item.year].customers = Math.max(yearGroups[item.year].customers, item.customers) // Use peak customers
    yearGroups[item.year].sales += item.sales
    yearGroups[item.year].profit += item.profit
    yearGroups[item.year].revenue += item.revenue
    yearGroups[item.year].months += 1
  })
  
  Object.values(yearGroups).forEach(yearData => {
    yearlyData.push({
      ...yearData,
      date: yearData.year.toString(),
      roi: Math.round((yearData.profit / yearData.sales) * 100 * 10) / 10,
    })
  })
  
  return yearlyData
}

const formatValue = (value) => {
  if (value >= 1000000) {
    return `$${(value / 1000000).toFixed(1)}M`
  } else if (value >= 1000) {
    return `$${(value / 1000).toFixed(1)}K`
  }
  return `$${value}`
}

const formatCustomers = (value) => {
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(1)}M`
  } else if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}K`
  }
  return value
}

export function BusinessImpactProjections() {
  const [timeframe, setTimeframe] = useState<'monthly' | 'yearly'>('yearly')
  const [metric, setMetric] = useState<'customers' | 'sales' | 'profit' | 'revenue' | 'roi'>('profit')
  const [showInteractive, setShowInteractive] = useState(false)
  
  const monthlyData = generateMonthlyData()
  const yearlyData = generateYearlyData(monthlyData)
  
  const data = timeframe === 'monthly' ? monthlyData : yearlyData
  
  const metricColors = {
    customers: '#8884d8',
    sales: '#82ca9d',
    profit: '#ff7300',
    revenue: '#0088fe',
    roi: '#ff5858'
  }
  
  const metricFormatters = {
    customers: (value) => formatCustomers(value),
    sales: (value) => formatValue(value),
    profit: (value) => formatValue(value),
    revenue: (value) => formatValue(value),
    roi: (value) => `${value}%`
  }
  
  const metricLabels = {
    customers: 'Customers',
    sales: 'Sales',
    profit: 'Profit',
    revenue: 'Revenue',
    roi: 'ROI'
  }
  
  return (
    <Card className="shadow-lg">
      <CardHeader>
        <div className="flex flex-col md:flex-row md:items-center justify-between">
          <div>
            <CardTitle>Business Impact Projections (2024-2034)</CardTitle>
            <CardDescription>ML automation impact on business growth metrics</CardDescription>
          </div>
          <div className="flex mt-4 md:mt-0 space-x-2">
            <Tabs defaultValue={timeframe} onValueChange={(v) => setTimeframe(v as 'monthly' | 'yearly')}>
              <TabsList>
                <TabsTrigger value="yearly">Yearly</TabsTrigger>
                <TabsTrigger value="monthly">Monthly</TabsTrigger>
              </TabsList>
            </Tabs>
            
            <Select value={metric} onValueChange={(v) => setMetric(v as 'customers' | 'sales' | 'profit' | 'revenue' | 'roi')}>
              <SelectTrigger className="w-[130px]">
                <SelectValue placeholder="Select metric" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="profit">Profit</SelectItem>
                <SelectItem value="revenue">Revenue</SelectItem>
                <SelectItem value="sales">Sales</SelectItem>
                <SelectItem value="customers">Customers</SelectItem>
                <SelectItem value="roi">ROI</SelectItem>
              </SelectContent>
            </Select>
            
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setShowInteractive(!showInteractive)}
            >
              {showInteractive ? "Simple View" : "Interactive View"}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {!showInteractive ? (
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={data}
                margin={{
                  top: 10,
                  right: 30,
                  left: 30,
                  bottom: 30,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis 
                  dataKey={timeframe === 'monthly' ? 'monthLabel' : 'year'}
                  tickFormatter={timeframe === 'monthly' ? ((tick, index) => index % 6 === 0 ? `${tick} ${data[index]?.year}` : '') : undefined}
                  interval={timeframe === 'monthly' ? 2 : 0}
                />
                <YAxis 
                  tickFormatter={(value) => metricFormatters[metric](value)}
                />
                <Tooltip
                  formatter={(value) => [metricFormatters[metric](value), metricLabels[metric]]}
                  labelFormatter={(label, items) => {
                    if (timeframe === 'monthly') {
                      const dataPoint = data.find(d => d.monthLabel === label || (d.month && d.month.toString() === label));
                      return dataPoint ? `${dataPoint.monthLabel} ${dataPoint.year}` : label;
                    }
                    return label;
                  }}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey={metric}
                  name={metricLabels[metric]}
                  fill={metricColors[metric]}
                  stroke={metricColors[metric]}
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={data}
                margin={{
                  top: 10,
                  right: 30,
                  left: 30,
                  bottom: 30,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis 
                  dataKey={timeframe === 'monthly' ? 'monthLabel' : 'year'} 
                  tickFormatter={timeframe === 'monthly' ? ((tick, index) => index % 6 === 0 ? `${tick} ${data[index]?.year}` : '') : undefined}
                  interval={timeframe === 'monthly' ? 2 : 0}
                />
                <YAxis 
                  yAxisId="left"
                  tickFormatter={(value) => metricFormatters[metric](value)}
                />
                <YAxis 
                  yAxisId="right" 
                  orientation="right"
                  tickFormatter={(value) => value >= 1000 ? `${(value/1000).toFixed(0)}K` : value}
                />
                <Tooltip
                  formatter={(value, name) => {
                    const formatter = metricFormatters[name.toLowerCase()] || (value => value);
                    return [formatter(value), name];
                  }}
                  labelFormatter={(label, items) => {
                    if (timeframe === 'monthly') {
                      const dataPoint = data.find(d => d.monthLabel === label || (d.month && d.month.toString() === label));
                      return dataPoint ? `${dataPoint.monthLabel} ${dataPoint.year}` : label;
                    }
                    return label;
                  }}
                />
                <Legend />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey={metric}
                  name={metricLabels[metric]}
                  fill={metricColors[metric]}
                  stroke={metricColors[metric]}
                  fillOpacity={0.3}
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="customers"
                  name="Customers"
                  stroke="#8884d8"
                  strokeWidth={2}
                  dot={timeframe === 'yearly'}
                />
                <Bar
                  yAxisId="left"
                  dataKey="roi"
                  name="ROI %"
                  fill="#ff5858"
                  radius={[4, 4, 0, 0]}
                  barSize={timeframe === 'monthly' ? 5 : 20}
                  opacity={0.7}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
        
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-blue-700 dark:text-blue-300">Projected 10-Year Revenue</h3>
            <p className="text-2xl font-bold">{formatValue(yearlyData.reduce((total, year) => total + year.revenue, 0))}</p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-green-700 dark:text-green-300">Projected 10-Year Profit</h3>
            <p className="text-2xl font-bold">{formatValue(yearlyData.reduce((total, year) => total + year.profit, 0))}</p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/30 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-purple-700 dark:text-purple-300">Customer Growth</h3>
            <p className="text-2xl font-bold">{formatCustomers(yearlyData[yearlyData.length - 1]?.customers - yearlyData[0]?.customers)} <span className="text-sm">({Math.round((yearlyData[yearlyData.length - 1]?.customers / yearlyData[0]?.customers - 1) * 100)}%)</span></p>
          </div>
          <div className="bg-amber-50 dark:bg-amber-900/30 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-amber-700 dark:text-amber-300">Average ROI</h3>
            <p className="text-2xl font-bold">{Math.round(yearlyData.reduce((total, year) => total + year.roi, 0) / yearlyData.length)}%</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
} 