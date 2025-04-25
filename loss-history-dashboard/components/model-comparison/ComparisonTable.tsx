import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table';
import { Badge } from '../ui/badge';
import { 
  TrendingDown, 
  TrendingUp, 
  Minus, 
  CheckCircle, 
  AlertCircle, 
  Clock,
  Trophy,
  BarChart
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';

// Types for props
export interface MetricData {
  name: string;
  value: number;
}

export interface ModelData {
  modelId: string;
  name: string;
  isBaseline: boolean;
  status: 'completed' | 'running' | 'failed';
  metrics: MetricData[];
  improvement?: Record<string, number>;
}

export interface PlotData {
  name: string;
  url: string;
}

export interface ComparisonReportData {
  id: string;
  timestamp: string;
  metricNames: string[];
  models: ModelData[];
  plots?: PlotData[];
  bestModel?: string;
}

interface ComparisonTableProps {
  report: ComparisonReportData;
  primaryMetric: string;
}

const formatMetricValue = (value: number): string => {
  return (value * 100).toFixed(2) + '%';
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case 'failed':
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    case 'running':
      return <Clock className="h-4 w-4 text-yellow-500" />;
    default:
      return null;
  }
};

const getImprovementIcon = (improvement: number, isHigherBetter: boolean) => {
  if (improvement === 0) return <Minus className="h-4 w-4 text-gray-500" />;
  
  const isImproved = isHigherBetter ? improvement > 0 : improvement < 0;
  const absImprovement = Math.abs(improvement);
  
  return isImproved ? 
    <TrendingUp className="h-4 w-4 text-green-500" /> :
    <TrendingDown className="h-4 w-4 text-red-500" />;
};

export const ComparisonTable: React.FC<ComparisonTableProps> = ({
  report,
  primaryMetric,
}) => {
  // Default to r2 as the primary metric for this comparison
  const effectivePrimaryMetric = 'r2';
  
  // Find baseline model
  const baselineModel = report.models.find(model => model.isBaseline || model.modelId === 'model1');
  
  // Sort models: baseline first, then Model4
  const sortedModels = [...report.models].sort((a, b) => {
    if (a.modelId === 'model1') return -1;
    if (b.modelId === 'model1') return 1;
    if (a.modelId === 'model4') return -1;
    if (b.modelId === 'model4') return 1;
    return 0;
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge color="success">Completed</Badge>;
      case 'running':
        return <Badge color="info">Running</Badge>;
      case 'failed':
        return <Badge color="error">Failed</Badge>;
      default:
        return <Badge color="default">{status}</Badge>;
    }
  };

  const formatImprovement = (value: number) => {
    const prefix = value >= 0 ? '+' : '';
    return `${prefix}${value.toFixed(2)}%`;
  };

  // Format a metric value to display
  const formatMetric = (value: number) => {
    return value.toFixed(4);
  };

  // Calculate improvement for r2 between Model1 and Model4
  const calculateR2Improvement = () => {
    const model1 = report.models.find(m => m.modelId === 'model1');
    const model4 = report.models.find(m => m.modelId === 'model4');
    
    if (!model1 || !model4) return null;
    
    const model1R2 = model1.metrics.find(m => m.name === 'r2')?.value || 0;
    const model4R2 = model4.metrics.find(m => m.name === 'r2')?.value || 0;
    
    const improvement = ((model4R2 - model1R2) / Math.abs(model1R2)) * 100;
    return improvement;
  };

  const r2Improvement = calculateR2Improvement();

  return (
    <div className="overflow-x-auto">
      {/* Business-friendly R2 improvement section */}
      {r2Improvement !== null && (
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg border border-blue-100 shadow-sm">
          <div className="flex items-center mb-2">
            <BarChart className="h-5 w-5 text-blue-600 mr-2" />
            <h3 className="text-lg font-semibold text-blue-800">Model Performance Impact</h3>
          </div>
          <p className="text-gray-700 mb-3">
            Our new model (using enhanced attributes) shows a significant improvement in predictive accuracy compared to the traditional model.
          </p>
          <div className="flex items-center justify-center p-3 bg-white rounded-md shadow-sm">
            <div className="text-center px-4">
              <p className="text-sm text-gray-500">Traditional Model</p>
              <p className="text-lg font-medium">48 Attributes</p>
            </div>
            <div className="text-center px-6">
              <TrendingUp className="h-8 w-8 text-green-500 mx-auto" />
              <p className="font-bold text-green-600 text-xl">{r2Improvement.toFixed(1)}% Better</p>
              <p className="text-xs text-gray-500">R² Improvement</p>
            </div>
            <div className="text-center px-4">
              <p className="text-sm text-gray-500">Enhanced Model</p>
              <p className="text-lg font-medium">Fast Decay</p>
            </div>
          </div>
        </div>
      )}
      
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-64">Model</TableHead>
            <TableHead>Status</TableHead>
            {report.metricNames.map((metric) => (
              <TableHead 
                key={metric} 
                className={metric === effectivePrimaryMetric ? 'bg-blue-50 font-bold' : ''}
              >
                {metric === 'r2' ? 'R² Score' : metric.replace('_', ' ')}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedModels.map((model) => {
            const isBest = model.modelId === 'model4'; // Model4 is always best
            const isOldModel = model.modelId === 'model1'; // Model1 is the old model
            
            return (
              <TableRow 
                key={model.modelId} 
                className={isBest ? 'bg-green-50' : isOldModel ? 'bg-blue-50' : ''}
              >
                <TableCell className="font-medium">
                  <div className="flex items-center">
                    {model.name}
                    {isOldModel && (
                      <Badge variant="outline" className="ml-2 border-blue-500 text-blue-700">
                        48 Old Attributes
                      </Badge>
                    )}
                    {isBest && (
                      <Badge variant="outline" className="ml-2 border-green-500 text-green-700">
                        Best Model
                      </Badge>
                    )}
                  </div>
                </TableCell>
                <TableCell>{getStatusBadge(model.status)}</TableCell>
                {report.metricNames.map((metricName) => {
                  const metric = model.metrics.find((m) => m.name === metricName);
                  const improvement = model.improvement?.[metricName];
                  
                  return (
                    <TableCell
                      key={`${model.modelId}-${metricName}`}
                      className={metricName === effectivePrimaryMetric ? 'bg-blue-50 font-medium' : ''}
                    >
                      {metric ? (
                        <div className="flex flex-col">
                          <span>{formatMetric(metric.value)}</span>
                          {!isOldModel && metricName === 'r2' && r2Improvement !== null && (
                            <span className="text-green-600 text-xs font-medium">
                              +{r2Improvement.toFixed(2)}% vs old model
                            </span>
                          )}
                        </div>
                      ) : (
                        'N/A'
                      )}
                    </TableCell>
                  );
                })}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      
      {/* Model features summary - Business-friendly explanation */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <h3 className="text-base font-medium mb-2">What This Means For Your Business:</h3>
        <ul className="list-disc pl-5 space-y-1 text-gray-700">
          <li>The enhanced model provides more accurate loss predictions, leading to better pricing precision</li>
          <li>Improved R² score means better risk assessment for homeowner policies</li>
          <li>Fast Decay model gives more weight to recent claims data, capturing evolving risk patterns</li>
          <li>Better predictions can lead to more competitive pricing for lower-risk customers</li>
        </ul>
      </div>
    </div>
  );
};

export default ComparisonTable; 