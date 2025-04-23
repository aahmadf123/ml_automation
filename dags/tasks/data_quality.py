import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from scipy import stats
import mlflow
from plugins.utils.slack import post as send_message

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'missing_threshold': 0.05,
            'outlier_threshold': 3.0,
            'drift_threshold': 0.1,
            'correlation_threshold': 0.7
        }
        self.history = []
        self.baseline_stats = None
        
    def calculate_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate basic statistics for each column."""
        stats = {}
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                stats[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'missing': df[column].isnull().mean(),
                    'unique_values': df[column].nunique()
                }
            else:
                stats[column] = {
                    'missing': df[column].isnull().mean(),
                    'unique_values': df[column].nunique(),
                    'most_common': df[column].value_counts().index[0] if not df[column].empty else None
                }
        return stats

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers using z-score method."""
        outliers = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outlier_indices = np.where(z_scores > self.config['outlier_threshold'])[0]
            if len(outlier_indices) > 0:
                outliers[column] = outlier_indices.tolist()
        return outliers

    def detect_drift(self, current_stats: Dict, baseline_stats: Dict) -> Dict:
        """Detect statistical drift between current and baseline data."""
        drift = {}
        for column in current_stats:
            if column in baseline_stats:
                current = current_stats[column]
                baseline = baseline_stats[column]
                
                if 'mean' in current and 'mean' in baseline:
                    mean_drift = abs(current['mean'] - baseline['mean']) / baseline['std']
                    if mean_drift > self.config['drift_threshold']:
                        drift[column] = {
                            'type': 'mean',
                            'current': current['mean'],
                            'baseline': baseline['mean'],
                            'drift': mean_drift
                        }
                
                if 'missing' in current and 'missing' in baseline:
                    missing_drift = abs(current['missing'] - baseline['missing'])
                    if missing_drift > self.config['missing_threshold']:
                        drift[column] = {
                            'type': 'missing',
                            'current': current['missing'],
                            'baseline': baseline['missing'],
                            'drift': missing_drift
                        }
        return drift

    def detect_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """Detect highly correlated features."""
        correlations = []
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i,j]
                    if abs(corr) > self.config['correlation_threshold']:
                        correlations.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr
                        })
        return correlations

    def monitor_data_quality(self, df: pd.DataFrame, is_baseline: bool = False) -> Dict:
        """Monitor data quality and detect issues."""
        current_time = datetime.now()
        
        # Calculate current statistics
        current_stats = self.calculate_basic_stats(df)
        
        # Store baseline if specified
        if is_baseline:
            self.baseline_stats = current_stats
            logger.info("Baseline statistics updated")
        
        # Initialize quality report
        quality_report = {
            'timestamp': current_time,
            'basic_stats': current_stats,
            'issues': []
        }
        
        # Check for missing values
        for column, stats in current_stats.items():
            if stats['missing'] > self.config['missing_threshold']:
                quality_report['issues'].append({
                    'type': 'missing_values',
                    'column': column,
                    'missing_rate': stats['missing'],
                    'threshold': self.config['missing_threshold']
                })
        
        # Detect outliers
        outliers = self.detect_outliers(df)
        if outliers:
            quality_report['issues'].append({
                'type': 'outliers',
                'details': outliers
            })
        
        # Detect drift if baseline exists
        if self.baseline_stats:
            drift = self.detect_drift(current_stats, self.baseline_stats)
            if drift:
                quality_report['issues'].append({
                    'type': 'drift',
                    'details': drift
                })
        
        # Detect correlations
        correlations = self.detect_correlations(df)
        if correlations:
            quality_report['issues'].append({
                'type': 'correlations',
                'details': correlations
            })
        
        # Store in history
        self.history.append(quality_report)
        
        # Log to MLflow
        mlflow.log_metrics({
            'missing_rate': sum(stats['missing'] for stats in current_stats.values()) / len(current_stats),
            'outlier_count': sum(len(indices) for indices in outliers.values()),
            'drift_count': len(drift) if 'drift' in quality_report else 0,
            'correlation_count': len(correlations)
        })
        
        # Alert on significant issues
        if quality_report['issues']:
            send_message(
                channel="#alerts",
                title="🔍 Data Quality Issues Detected",
                details="\n".join([
                    f"{issue['type']}: {len(issue.get('details', []))} issues found"
                    for issue in quality_report['issues']
                ]),
                urgency="medium"
            )
        
        return quality_report

    def get_quality_summary(self, window_hours: int = 24) -> Dict:
        """Get a summary of data quality metrics over the specified time window."""
        if not self.history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_history = [h for h in self.history if h['timestamp'] >= cutoff_time]
        
        if not recent_history:
            return {}
        
        # Aggregate issues
        issue_counts = {}
        for report in recent_history:
            for issue in report['issues']:
                issue_type = issue['type']
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        return {
            'total_reports': len(recent_history),
            'issue_counts': issue_counts,
            'latest_report': recent_history[-1],
            'window_hours': window_hours
        } 