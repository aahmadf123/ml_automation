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
  Trophy
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
  // Find baseline model
  const baselineModel = report.models.find(model => model.isBaseline);
  
  // Sort models: baseline first, then by primary metric (descending)
  const sortedModels = [...report.models].sort((a, b) => {
    if (a.isBaseline) return -1;
    if (b.isBaseline) return 1;
    
    const aMetric = a.metrics.find(m => m.name === primaryMetric)?.value || 0;
    const bMetric = b.metrics.find(m => m.name === primaryMetric)?.value || 0;
    
    return bMetric - aMetric; // Descending order
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

  return (
    <div className="overflow-x-auto">
      <Table>
        <thead>
          <tr>
            <th className="w-64">Model</th>
            <th>Status</th>
            {report.metricNames.map((metric) => (
              <th key={metric} className={metric === primaryMetric ? 'bg-blue-50 font-bold' : ''}>
                {metric.replace('_', ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedModels.map((model) => {
            const isBest = report.bestModel === model.modelId;
            return (
              <tr key={model.modelId} className={isBest ? 'bg-green-50' : ''}>
                <td className="font-medium">
                  {model.name}
                  {model.isBaseline && (
                    <Badge color="secondary" className="ml-2">
                      Baseline
                    </Badge>
                  )}
                  {isBest && (
                    <Badge color="success" className="ml-2">
                      Best
                    </Badge>
                  )}
                </td>
                <td>{getStatusBadge(model.status)}</td>
                {report.metricNames.map((metricName) => {
                  const metric = model.metrics.find((m) => m.name === metricName);
                  const improvement = model.improvement?.[metricName];
                  
                  return (
                    <td 
                      key={`${model.modelId}-${metricName}`}
                      className={metricName === primaryMetric ? 'bg-blue-50' : ''}
                    >
                      {metric ? (
                        <div className="flex flex-col">
                          <span>{formatMetric(metric.value)}</span>
                          {!model.isBaseline && improvement !== undefined && (
                            <Tooltip content={`Improvement over baseline`}>
                              <span className={improvement >= 0 ? 'text-green-600 text-xs' : 'text-red-600 text-xs'}>
                                {formatImprovement(improvement)}
                              </span>
                            </Tooltip>
                          )}
                        </div>
                      ) : (
                        'N/A'
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </Table>
    </div>
  );
};

export default ComparisonTable; 