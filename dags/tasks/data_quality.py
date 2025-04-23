import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from scipy import stats
import mlflow
import gc

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
        """Calculate basic statistics for each column with memory optimization."""
        stats = {}
        # Process in smaller batches of columns
        column_batches = [df.columns[i:i+10] for i in range(0, len(df.columns), 10)]
        
        for batch in column_batches:
            for column in batch:
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Calculate one statistic at a time
                    stats[column] = {
                        'mean': float(df[column].mean()),  # Convert numpy types to Python natives
                        'std': float(df[column].std()),
                        'min': float(df[column].min()),
                        'max': float(df[column].max()),
                        'missing': float(df[column].isnull().mean()),
                        'unique_values': int(df[column].nunique())
                    }
                else:
                    stats[column] = {
                        'missing': float(df[column].isnull().mean()),
                        'unique_values': int(df[column].nunique()),
                        'most_common': str(df[column].value_counts().index[0]) if not df[column].empty else None
                    }
                # Clear column from memory
                gc.collect()
                
        return stats
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict:
        """Detect missing values above threshold with memory optimization."""
        missing = {}
        # Process in smaller batches of columns
        column_batches = [df.columns[i:i+20] for i in range(0, len(df.columns), 20)]
        
        for batch in column_batches:
            for column in batch:
                missing_rate = df[column].isnull().mean()
                if missing_rate > self.config['missing_threshold']:
                    missing[column] = float(missing_rate)  # Convert to native Python float
            gc.collect()  # Clear memory after each batch
        
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
        """Detect outliers using z-score with memory optimization."""
        outliers = {}
        # Process numeric columns in batches
        numeric_cols = df.select_dtypes(include=['number']).columns
        column_batches = [numeric_cols[i:i+10] for i in range(0, len(numeric_cols), 10)]
        
        for batch in column_batches:
            for column in batch:
                # Calculate z-scores in chunks to avoid large arrays
                chunk_size = 10000
                n_chunks = len(df) // chunk_size + 1
                outlier_count = 0
                
                mean_val = df[column].mean()
                std_val = df[column].std()
                
                if std_val == 0:  # Skip if standard deviation is zero (constant column)
                    continue
                    
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(df))
                    chunk = df[column].iloc[start_idx:end_idx]
                    
                    # Calculate z-scores and count outliers
                    z_scores = np.abs((chunk - mean_val) / std_val)
                    chunk_outliers = (z_scores > self.config['outlier_threshold']).sum()
                    outlier_count += chunk_outliers
                    
                    # Clean up
                    del z_scores
                    gc.collect()
                
                if outlier_count > 0:
                    outliers[column] = float(outlier_count / len(df))
                
            # Clear memory after each batch
            gc.collect()
        
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
        """Detect drift in numerical columns with memory optimization."""
        if self.baseline_stats is None:
            logger.warning("No baseline stats available for drift detection")
            return {}
        
        drift = {}
        # Process in smaller batches
        column_batches = [df.columns[i:i+20] for i in range(0, len(df.columns), 20)]
        
        for batch in column_batches:
            for column in batch:
                if column in self.baseline_stats and pd.api.types.is_numeric_dtype(df[column]):
                    current_mean = float(df[column].mean())
                    baseline_mean = self.baseline_stats[column]['mean']
                    
                    if baseline_mean != 0:
                        drift_pct = abs(current_mean - baseline_mean) / baseline_mean
                        if drift_pct > self.config['drift_threshold']:
                            drift[column] = float(drift_pct)
            
            # Clear memory after each batch
            gc.collect()
        
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
        """Detect changes in correlation between numeric columns with memory optimization."""
        if self.baseline_stats is None or 'correlation' not in self.baseline_stats:
            logger.warning("No baseline correlation available")
            return {}
        
        # Limit to a subset of numeric columns to reduce memory usage
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # If too many numeric columns, sample them
        if len(numeric_cols) > 30:
            logger.info(f"Limiting correlation analysis to 30 columns out of {len(numeric_cols)}")
            numeric_cols = numeric_cols[:30]  # Take first 30 columns
            
        current_corr = df[numeric_cols].corr()
        baseline_corr = self.baseline_stats['correlation']
        
        correlation_changes = {}
        # Compare correlations
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:  # Only check each pair once
                if col1 in baseline_corr and col2 in baseline_corr:
                    try:
                        current = current_corr.loc[col1, col2]
                        baseline = baseline_corr.loc[col1, col2]
                        change = abs(current - baseline)
                        
                        if change > self.config['correlation_threshold']:
                            correlation_changes[f"{col1} ↔ {col2}"] = float(change)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error comparing correlation for {col1} and {col2}: {e}")
        
        # Clear memory
        del current_corr
        gc.collect()
        
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
        
        # Calculate correlation for a limited subset of columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        # If too many numeric columns, sample them
        if len(numeric_cols) > 30:
            logger.info(f"Limiting correlation baseline to 30 columns out of {len(numeric_cols)}")
            numeric_cols = numeric_cols[:30]
            
        stats['correlation'] = df[numeric_cols].corr()
        self.baseline_stats = stats
        logger.info("Baseline statistics set")
        
        # Clear memory
        gc.collect()
        
    def run_quality_checks(self, df: pd.DataFrame) -> Dict:
        """Run all quality checks on the dataframe with memory optimization."""
        # Run each check separately with garbage collection in between
        logger.info("Running missing value detection")
        missing_values = self.detect_missing_values(df)
        gc.collect()
        
        logger.info("Running outlier detection")
        outliers = self.detect_outliers(df)
        gc.collect()
        
        logger.info("Running data drift detection")
        data_drift = self.detect_data_drift(df)
        gc.collect()
        
        logger.info("Running correlation change detection")
        correlation_changes = self.detect_correlation_changes(df)
        gc.collect()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'missing_values': missing_values,
            'outliers': outliers,
            'data_drift': data_drift,
            'correlation_changes': correlation_changes
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