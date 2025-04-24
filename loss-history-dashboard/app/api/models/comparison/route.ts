import { NextResponse } from 'next/server'
import { ComparisonReportData } from '../../../../components/model-comparison/ComparisonTable'

// Mock data for development purposes
// In production, this would fetch from your backend API or S3
const generateMockComparisonReports = () => {
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
  
  // Generate 3 comparison reports with different timestamps
  return Array.from({ length: 3 }).map((_, reportIndex) => {
    const reportId = `report-${reportIndex + 1}`
    const timestamp = new Date(Date.now() - reportIndex * 7 * 24 * 60 * 60 * 1000).toISOString() // each a week apart
    
    // Generate 3-5 models for each report
    const modelCount = 3 + Math.floor(Math.random() * 3)
    const models = Array.from({ length: modelCount }).map((_, modelIndex) => {
      const modelId = `model-${reportIndex}-${modelIndex + 1}`
      const isBaseline = modelIndex === 0
      
      // Generate metrics for each model
      const modelMetrics = metrics.map(metricName => {
        const baseValue = metricName === 'accuracy' ? 0.85 : 
                         metricName === 'precision' ? 0.82 : 
                         metricName === 'recall' ? 0.78 : 
                         metricName === 'f1_score' ? 0.80 : 
                         metricName === 'auc' ? 0.88 : 0.75
                         
        // Add some variation between models
        const variation = (Math.random() * 0.2) - 0.1
        
        return {
          name: metricName,
          value: Math.min(1, Math.max(0, baseValue + variation)),
          isHigherBetter: true
        }
      })
      
      // Calculate improvement over baseline for non-baseline models
      const improvement = !isBaseline ? 
        metrics.reduce((acc, metric) => {
          const modelMetric = modelMetrics.find(m => m.name === metric)?.value || 0
          const baselineModel = models[0]
          const baselineMetric = baselineModel?.metrics.find(m => m.name === metric)?.value || 0
          
          if (baselineMetric > 0) {
            const percentChange = ((modelMetric - baselineMetric) / baselineMetric) * 100
            acc[metric] = parseFloat(percentChange.toFixed(2))
          }
          
          return acc
        }, {} as Record<string, number>) : undefined
      
      return {
        modelId,
        name: `XGBoost ${isBaseline ? 'Baseline' : ''}`,
        metrics: modelMetrics,
        status: 'completed',
        timestamp: new Date(Date.now() - (reportIndex * 7 + modelIndex) * 24 * 60 * 60 * 1000).toISOString(),
        isBaseline,
        improvement
      }
    })
    
    // Determine the best model based on f1_score
    const bestModelIndex = models.reduce((bestIdx, model, currentIdx) => {
      const currentF1 = model.metrics.find(m => m.name === 'f1_score')?.value || 0
      const bestF1 = models[bestIdx].metrics.find(m => m.name === 'f1_score')?.value || 0
      return currentF1 > bestF1 ? currentIdx : bestIdx
    }, 0)
    
    // Generate mock plot URLs
    const plots = [
      { name: 'feature_importance', url: '/mockdata/feature_importance.png' },
      { name: 'confusion_matrix', url: '/mockdata/confusion_matrix.png' },
      { name: 'roc_curve', url: '/mockdata/roc_curve.png' },
      { name: 'precision_recall_curve', url: '/mockdata/precision_recall.png' }
    ]
    
    return {
      id: reportId,
      timestamp,
      models,
      bestModel: models[bestModelIndex].modelId,
      metricNames: metrics,
      plots
    }
  })
}

export async function GET() {
  try {
    // This would be replaced with actual data fetching from your backend
    const mockData: ComparisonReportData[] = [
      {
        id: "report-1",
        timestamp: new Date().toISOString(),
        metricNames: ["accuracy", "precision", "recall", "f1_score", "auc"],
        models: [
          {
            modelId: "model-1",
            name: "XGBoost Baseline",
            isBaseline: true,
            status: "completed",
            metrics: [
              { name: "accuracy", value: 0.86 },
              { name: "precision", value: 0.84 },
              { name: "recall", value: 0.82 },
              { name: "f1_score", value: 0.83 },
              { name: "auc", value: 0.91 }
            ]
          },
          {
            modelId: "model-2",
            name: "XGBoost Equal Weight",
            isBaseline: false,
            status: "completed",
            metrics: [
              { name: "accuracy", value: 0.89 },
              { name: "precision", value: 0.87 },
              { name: "recall", value: 0.86 },
              { name: "f1_score", value: 0.865 },
              { name: "auc", value: 0.93 }
            ],
            improvement: {
              "accuracy": 3.5,
              "precision": 3.6,
              "recall": 4.9,
              "f1_score": 4.2,
              "auc": 2.2
            }
          },
          {
            modelId: "model-3",
            name: "XGBoost Linear Decay",
            isBaseline: false,
            status: "completed",
            metrics: [
              { name: "accuracy", value: 0.92 },
              { name: "precision", value: 0.9 },
              { name: "recall", value: 0.89 },
              { name: "f1_score", value: 0.895 },
              { name: "auc", value: 0.95 }
            ],
            improvement: {
              "accuracy": 7.0,
              "precision": 7.1,
              "recall": 8.5,
              "f1_score": 7.8,
              "auc": 4.4
            }
          },
          {
            modelId: "model-4",
            name: "XGBoost Fast Decay",
            isBaseline: false,
            status: "running",
            metrics: [
              { name: "accuracy", value: 0.88 },
              { name: "precision", value: 0.86 },
              { name: "recall", value: 0.84 },
              { name: "f1_score", value: 0.85 },
              { name: "auc", value: 0.92 }
            ],
            improvement: {
              "accuracy": 2.3,
              "precision": 2.4,
              "recall": 2.4,
              "f1_score": 2.4,
              "auc": 1.1
            }
          },
          {
            modelId: "model-5",
            name: "LightGBM (experimental)",
            isBaseline: false,
            status: "failed",
            metrics: [
              { name: "accuracy", value: 0.82 },
              { name: "precision", value: 0.8 },
              { name: "recall", value: 0.78 },
              { name: "f1_score", value: 0.79 },
              { name: "auc", value: 0.87 }
            ],
            improvement: {
              "accuracy": -4.7,
              "precision": -4.8,
              "recall": -4.9,
              "f1_score": -4.8,
              "auc": -4.4
            }
          }
        ],
        plots: [
          {
            name: "ROC Curve Comparison",
            url: "https://via.placeholder.com/800x600.png?text=ROC+Curve+Comparison"
          },
          {
            name: "Precision-Recall Curves",
            url: "https://via.placeholder.com/800x600.png?text=Precision-Recall+Curves"
          },
          {
            name: "Feature Importance Comparison",
            url: "https://via.placeholder.com/800x600.png?text=Feature+Importance+Comparison"
          },
          {
            name: "Learning Curves",
            url: "https://via.placeholder.com/800x600.png?text=Learning+Curves"
          }
        ],
        bestModel: "model-3"
      },
      {
        id: "report-2",
        timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(), // 1 week ago
        metricNames: ["accuracy", "precision", "recall", "f1_score"],
        models: [
          {
            modelId: "model-old-1",
            name: "XGBoost Baseline",
            isBaseline: true,
            status: "completed",
            metrics: [
              { name: "accuracy", value: 0.84 },
              { name: "precision", value: 0.82 },
              { name: "recall", value: 0.8 },
              { name: "f1_score", value: 0.81 }
            ]
          },
          {
            modelId: "model-old-2",
            name: "XGBoost Equal Weight",
            isBaseline: false,
            status: "completed",
            metrics: [
              { name: "accuracy", value: 0.86 },
              { name: "precision", value: 0.85 },
              { name: "recall", value: 0.83 },
              { name: "f1_score", value: 0.84 }
            ],
            improvement: {
              "accuracy": 2.4,
              "precision": 3.7,
              "recall": 3.8,
              "f1_score": 3.7
            }
          },
          {
            modelId: "model-old-3",
            name: "XGBoost Fast Decay",
            isBaseline: false,
            status: "completed",
            metrics: [
              { name: "accuracy", value: 0.89 },
              { name: "precision", value: 0.87 },
              { name: "recall", value: 0.86 },
              { name: "f1_score", value: 0.865 }
            ],
            improvement: {
              "accuracy": 6.0,
              "precision": 6.1,
              "recall": 7.5,
              "f1_score": 6.8
            }
          }
        ],
        plots: [
          {
            name: "ROC Curve Comparison",
            url: "https://via.placeholder.com/800x600.png?text=Old+ROC+Curve+Comparison"
          },
          {
            name: "Precision-Recall Curves",
            url: "https://via.placeholder.com/800x600.png?text=Old+Precision-Recall+Curves"
          }
        ],
        bestModel: "model-old-3"
      }
    ];

    return NextResponse.json(mockData);
  } catch (error) {
    console.error('Error fetching model comparison data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch model comparison data' },
      { status: 500 }
    );
  }
} 