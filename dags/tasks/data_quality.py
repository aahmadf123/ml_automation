import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from scipy import stats
import mlflow

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
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict:
        """Detect missing values above threshold."""
        missing = {}
        for column in df.columns:
            missing_rate = df[column].isnull().mean()
            if missing_rate > self.config['missing_threshold']:
                missing[column] = missing_rate
        
        if missing:
            # Import slack only when needed
            from utils.slack import post as send_message
            send_message(
                channel="#alerts",
                title="🔍 Missing Data Alert",
                details=f"Missing data detected in {len(missing)} columns:\n" +
                        "\n".join([f"{col}: {rate:.2%}" for col, rate in missing.items()]),
                urgency="high"
            )
            
        return missing
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using z-score."""
        outliers = {}
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outlier_indices = np.where(z_scores > self.config['outlier_threshold'])[0]
                if len(outlier_indices) > 0:
                    outliers[column] = len(outlier_indices) / len(df)
        
        if outliers:
            # Import slack only when needed
            from utils.slack import post as send_message
            send_message(
                channel="#alerts",
                title="⚠️ Outlier Alert",
                details=f"Outliers detected in {len(outliers)} columns:\n" +
                        "\n".join([f"{col}: {rate:.2%}" for col, rate in outliers.items()]),
                urgency="medium"
            )
            
        return outliers
    
    def detect_data_drift(self, df: pd.DataFrame) -> Dict:
        """Detect drift in numerical columns."""
        if self.baseline_stats is None:
            logger.warning("No baseline stats available for drift detection")
            return {}
        
        drift = {}
        for column in df.columns:
            if column in self.baseline_stats and pd.api.types.is_numeric_dtype(df[column]):
                current_mean = df[column].mean()
                baseline_mean = self.baseline_stats[column]['mean']
                
                if baseline_mean != 0:
                    drift_pct = abs(current_mean - baseline_mean) / baseline_mean
                    if drift_pct > self.config['drift_threshold']:
                        drift[column] = drift_pct
        
        if drift:
            # Import slack only when needed
            from utils.slack import post as send_message
            send_message(
                channel="#alerts",
                title="🚨 Data Drift Detected",
                details=f"Drift detected in {len(drift)} columns:\n" +
                        "\n".join([f"{col}: {drift_pct:.2%}" for col, drift_pct in drift.items()]),
                urgency="critical"
            )
            
        return drift
    
    def detect_correlation_changes(self, df: pd.DataFrame) -> Dict:
        """Detect changes in correlation between numeric columns."""
        if self.baseline_stats is None or 'correlation' not in self.baseline_stats:
            logger.warning("No baseline correlation available")
            return {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        current_corr = df[numeric_cols].corr()
        baseline_corr = self.baseline_stats['correlation']
        
        correlation_changes = {}
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 != col2 and col1 in baseline_corr and col2 in baseline_corr:
                    current = current_corr.loc[col1, col2]
                    baseline = baseline_corr.loc[col1, col2]
                    change = abs(current - baseline)
                    
                    if change > self.config['correlation_threshold']:
                        correlation_changes[f"{col1} ↔ {col2}"] = change
        
        if correlation_changes:
            # Import slack only when needed
            from utils.slack import post as send_message
            send_message(
                channel="#alerts",
                title="🔄 Correlation Changes Detected",
                details=f"Changes in {len(correlation_changes)} correlations:\n" +
                        "\n".join([f"{cols}: {change:.2f}" for cols, change in correlation_changes.items()]),
                urgency="medium"
            )
            
        return correlation_changes
    
    def set_baseline(self, df: pd.DataFrame) -> None:
        """Set baseline statistics for drift detection."""
        stats = self.calculate_basic_stats(df)
        stats['correlation'] = df.select_dtypes(include=['number']).corr()
        self.baseline_stats = stats
        logger.info("Baseline statistics set")
        
    def run_quality_checks(self, df: pd.DataFrame) -> Dict:
        """Run all quality checks on the dataframe."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'missing_values': self.detect_missing_values(df),
            'outliers': self.detect_outliers(df),
            'data_drift': self.detect_data_drift(df),
            'correlation_changes': self.detect_correlation_changes(df)
        }
        
        self.history.append(results)
        
        # Log the results and determine overall status
        issues = sum(len(check) for check in results.values() if isinstance(check, dict))
        if issues == 0:
            status = "pass"
            logger.info("Data quality checks passed")
        else:
            status = "fail"
            logger.warning(f"Data quality checks found {issues} issues")
            
        results['status'] = status
        return results
    
    def export_report(self, filepath: str) -> None:
        """Export data quality history to a file."""
        try:
            pd.DataFrame(self.history).to_csv(filepath, index=False)
            logger.info(f"Data quality report exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export data quality report: {str(e)}")
            
def manual_override(model_id: str, override_action: str) -> Dict:
    """Manual override function for Airflow DAG."""
    logger.info(f"Manual override requested for {model_id}: {override_action}")
    
    # Import slack only when needed
    from utils.slack import post as send_message
    send_message(
        channel="#alerts",
        title="🔧 Manual Override",
        details=f"Manual override applied to {model_id}:\n{override_action}",
        urgency="high"
    )
    
    return {
        "status": "success",
        "model_id": model_id,
        "action": override_action,
        "timestamp": datetime.now().isoformat()
    } 