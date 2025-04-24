import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from scipy import stats
import mlflow
import gc
from utils.cache import GLOBAL_CACHE, cache_result
import os

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
        # Use the global cache instead of recalculating
        df_name = f"data_quality_{id(df)}"
        return GLOBAL_CACHE.compute_statistics(df, df_name)
    
    @cache_result
    def detect_missing_values(self, df: pd.DataFrame) -> Dict:
        """Detect missing values above threshold with memory optimization."""
        missing = {}
        
        # Get cached statistics if available
        df_name = f"data_quality_{id(df)}"
        stats = GLOBAL_CACHE.statistics_cache.get(df_name)
        
        # If stats are already computed, use them
        if stats:
            for column, col_stats in stats.items():
                if 'missing_pct' in col_stats and col_stats['missing_pct'] > self.config['missing_threshold']:
                    missing[column] = float(col_stats['missing_pct'])
        else:
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
    
    @cache_result
    def detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using z-score with memory optimization."""
        outliers = {}
        try:
            # Process numeric columns in batches
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # For large datasets, sample the data for outlier detection
            sample_rows = None
            if df.shape[0] > 100000:
                sample_rows = 100000
                logger.info(f"Using {sample_rows} rows sample for outlier detection")
                
            # Use the parallel processing function from cache module
            def process_col(df, col):
                try:
                    # For large dataframes, use a sample
                    if sample_rows and df.shape[0] > sample_rows:
                        df_sample = df[[col]].sample(sample_rows, random_state=42)
                    else:
                        df_sample = df[[col]]
                    
                    mean_val = GLOBAL_CACHE.get_statistic(f"data_quality_{id(df)}", col, 'mean')
                    std_val = GLOBAL_CACHE.get_statistic(f"data_quality_{id(df)}", col, 'std')
                    
                    if mean_val is None or std_val is None or std_val == 0:
                        # Calculate if not in cache or std is zero (constant column)
                        mean_val = df_sample[col].mean()
                        std_val = df_sample[col].std()
                        if std_val == 0:
                            return 0
                            
                    # Calculate z-scores in chunks to avoid large arrays
                    chunk_size = min(10000, len(df_sample))
                    n_chunks = len(df_sample) // chunk_size + 1
                    outlier_count = 0
                    
                    for i in range(n_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(df_sample))
                        chunk = df_sample[col].iloc[start_idx:end_idx]
                        
                        # Calculate z-scores and count outliers
                        z_scores = np.abs((chunk - mean_val) / std_val)
                        chunk_outliers = (z_scores > self.config['outlier_threshold']).sum()
                        outlier_count += chunk_outliers
                        
                        # Clean up
                        del z_scores
                        
                    return outlier_count / len(df_sample) if outlier_count > 0 else 0
                except Exception as e:
                    logger.warning(f"Error processing outliers for column {col}: {str(e)}")
                    return 0
                    
            # Use parallel processing from the cache module with timeout
            max_workers = min(os.cpu_count() or 4, 4)  # Limit workers based on CPUs available
            if len(numeric_cols) > 10:
                col_results = GLOBAL_CACHE.parallel_column_processing(
                    df, process_col, numeric_cols, n_workers=max_workers
                )
                outliers = {col: rate for col, rate in col_results.items() if rate > 0}
            else:
                for col in numeric_cols:
                    rate = process_col(df, col)
                    if rate > 0:
                        outliers[col] = float(rate)
            
            if outliers:
                # Import slack only when needed
                try:
                    from utils.slack import post as send_message
                    send_message(
                        channel="#alerts",
                        title="⚠️ Outlier Alert",
                        details=f"Outliers detected in {len(outliers)} columns:\n" +
                                "\n".join([f"{col}: {rate:.2%}" for col, rate in outliers.items()]),
                        urgency="medium"
                    )
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {e}")
                
            return outliers
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            # Return empty dict instead of failing the pipeline
            return {}
    
    @cache_result
    def detect_data_drift(self, df: pd.DataFrame) -> Dict:
        """Detect drift in numerical columns with memory optimization."""
        if self.baseline_stats is None:
            logger.warning("No baseline stats available for drift detection")
            return {}
        
        drift = {}
        # Get cached statistics for current data
        df_name = f"data_quality_{id(df)}"
        current_stats = GLOBAL_CACHE.statistics_cache.get(df_name)
        
        # Process in smaller batches with cached statistics if available
        if current_stats:
            for column, stats in current_stats.items():
                if column in self.baseline_stats and 'mean' in stats:
                    current_mean = stats['mean']
                    baseline_mean = self.baseline_stats[column]['mean']
                    
                    if baseline_mean != 0:
                        drift_pct = abs(current_mean - baseline_mean) / baseline_mean
                        if drift_pct > self.config['drift_threshold']:
                            drift[column] = float(drift_pct)
        else:
            # Fallback to direct calculation
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
    
    @cache_result
    def detect_correlation_changes(self, df: pd.DataFrame) -> Dict:
        """Detect changes in correlation between numeric columns with memory optimization."""
        if self.baseline_stats is None or 'correlation' not in self.baseline_stats:
            logger.warning("No baseline correlation available")
            return {}
        
        # Limit to a subset of numeric columns to reduce memory usage
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Further limit columns for very large datasets
        if len(numeric_cols) > 30:
            logger.info(f"Limiting correlation analysis to 30 columns out of {len(numeric_cols)}")
            numeric_cols = numeric_cols[:30]  # Take first 30 columns
        elif len(numeric_cols) > 10 and df.shape[0] > 1000000:
            # For very large datasets, restrict even more
            logger.info(f"Large dataset detected ({df.shape[0]} rows). Limiting to 10 columns.")
            numeric_cols = numeric_cols[:10]
            
        try:
            # Compute correlations with caching and sample to reduce memory usage
            # For very large datasets, use a sample
            sample_size = min(100000, df.shape[0])
            if df.shape[0] > 200000:
                logger.info(f"Using {sample_size} rows sample for correlation calculation")
                df_sample = df[numeric_cols].sample(sample_size, random_state=42)
                df_name = f"data_quality_sample_{id(df)}"
                current_corr = GLOBAL_CACHE.compute_correlations(df_sample, df_name)
            else:
                df_name = f"data_quality_{id(df)}"
                current_corr = GLOBAL_CACHE.compute_correlations(df[numeric_cols], df_name)
                
            baseline_corr = self.baseline_stats['correlation']
            
            correlation_changes = {}
            # Compare correlations
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:  # Only check each pair once
                    if col1 in baseline_corr and col2 in baseline_corr and col1 in current_corr.index and col2 in current_corr.columns:
                        try:
                            current = current_corr.loc[col1, col2]
                            baseline = baseline_corr.loc[col1, col2]
                            
                            # Handle NaN values
                            if pd.isna(current) or pd.isna(baseline):
                                continue
                                
                            change = abs(current - baseline)
                            
                            if change > self.config['correlation_threshold']:
                                correlation_changes[f"{col1} ↔ {col2}"] = float(change)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Error comparing correlation for {col1} and {col2}: {e}")
            
            # Clear memory
            gc.collect()
            
            if correlation_changes:
                # Import slack only when needed
                try:
                    from utils.slack import post as send_message
                    send_message(
                        channel="#alerts",
                        title="🔄 Correlation Changes Detected",
                        details=f"Correlation changes detected in {len(correlation_changes)} pairs:\n" +
                                "\n".join([f"{pair}: {change:.2f}" for pair, change in correlation_changes.items()]),
                        urgency="medium"
                    )
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {e}")
                
            return correlation_changes
            
        except Exception as e:
            logger.error(f"Error in correlation change detection: {str(e)}")
            # Return empty dict instead of failing
            return {}
    
    def set_baseline(self, df: pd.DataFrame) -> None:
        """Set baseline statistics for drift detection."""
        df_name = f"baseline_{id(df)}"
        stats = GLOBAL_CACHE.compute_statistics(df, df_name)
        
        # Calculate correlation for a limited subset of columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        # If too many numeric columns, sample them
        if len(numeric_cols) > 30:
            logger.info(f"Limiting correlation baseline to 30 columns out of {len(numeric_cols)}")
            numeric_cols = numeric_cols[:30]
        
        stats['correlation'] = GLOBAL_CACHE.compute_correlations(df[numeric_cols], df_name)
        self.baseline_stats = stats
        logger.info("Baseline statistics set")
        
        # Clear memory
        gc.collect()
        
    def run_quality_checks(self, df: pd.DataFrame) -> Dict:
        """Run all quality checks on the dataframe with memory optimization."""
        # Initialize result dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'missing_values': {},
            'outliers': {},
            'data_drift': {},
            'correlation_changes': {},
            'total_issues': 0,
            'missing_value_issues': 0,
            'outlier_issues': 0
        }
        
        try:
            # Compute statistics once and store in cache
            df_name = f"data_quality_{id(df)}"
            GLOBAL_CACHE.compute_statistics(df, df_name)
            
            # Run each check separately with garbage collection in between
            # Wrap each check in try-except to ensure pipeline continues even if one check fails
            try:
                logger.info("Running missing value detection")
                missing_values = self.detect_missing_values(df)
                results['missing_values'] = missing_values
                results['missing_value_issues'] = len(missing_values)
                gc.collect()
            except Exception as e:
                logger.error(f"Error in missing value detection: {str(e)}")
                results['missing_values'] = {}
                results['missing_value_issues'] = 0
            
            try:
                logger.info("Running outlier detection")
                outliers = self.detect_outliers(df)
                results['outliers'] = outliers
                results['outlier_issues'] = len(outliers)
                gc.collect()
            except Exception as e:
                logger.error(f"Error in outlier detection: {str(e)}")
                results['outliers'] = {}
                results['outlier_issues'] = 0
            
            try:
                logger.info("Running data drift detection")
                data_drift = self.detect_data_drift(df)
                results['data_drift'] = data_drift
                gc.collect()
            except Exception as e:
                logger.error(f"Error in data drift detection: {str(e)}")
                results['data_drift'] = {}
            
            try:
                logger.info("Running correlation change detection")
                correlation_changes = self.detect_correlation_changes(df)
                results['correlation_changes'] = correlation_changes
                gc.collect()
            except Exception as e:
                logger.error(f"Error in correlation change detection: {str(e)}")
                results['correlation_changes'] = {}
            
            # Calculate total issues
            issues = sum(len(check) for check in [
                results['missing_values'], 
                results['outliers'], 
                results['data_drift'], 
                results['correlation_changes']
            ])
            results['total_issues'] = issues
            
            self.history.append(results)
            
            # Log the results and determine overall status
            # Import slack only when needed
            if issues > 0:
                try:
                    from utils.slack import post as send_message
                    summary = f"Data quality checks completed with {issues} issues detected:\n"
                    if results['missing_values']:
                        summary += f"- Missing values: {len(results['missing_values'])} columns\n"
                    if results['outliers']:
                        summary += f"- Outliers: {len(results['outliers'])} columns\n"
                    if results['data_drift']:
                        summary += f"- Data drift: {len(results['data_drift'])} columns\n"
                    if results['correlation_changes']:
                        summary += f"- Correlation changes: {len(results['correlation_changes'])} pairs\n"
                        
                    send_message(
                        channel="#data-quality",
                        title="📊 Data Quality Report",
                        details=summary,
                        urgency="high" if issues > 5 else "medium"
                    )
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in run_quality_checks: {str(e)}")
            # Return a minimal valid result to avoid pipeline failure
            results['error'] = str(e)
            return results
    
    def export_report(self, filepath: str) -> None:
        """Export a data quality report to a file."""
        if not self.history:
            logger.warning("No history to export")
            return
            
        with open(filepath, 'w') as f:
            import json
            json.dump(self.history[-1], f, indent=2)
        
        logger.info(f"Data quality report exported to {filepath}")


def manual_override(model_id: str, override_action: str) -> Dict:
    """
    Apply manual overrides for data quality alerts.
    
    Args:
        model_id: Model identifier
        override_action: Action to take (e.g., "ignore_drift", "ignore_missing")
        
    Returns:
        Status report dict
    """
    from airflow.models import Variable
    import json
    
    # Get current overrides
    try:
        overrides = json.loads(Variable.get(f"DQ_OVERRIDES_{model_id}", default_var="{}"))
    except (ValueError, TypeError):
        overrides = {}
    
    # Update overrides
    timestamp = datetime.now().isoformat()
    overrides[override_action] = {
        "timestamp": timestamp,
        "expires": (datetime.now() + timedelta(days=7)).isoformat()
    }
    
    # Save back to variable
    Variable.set(f"DQ_OVERRIDES_{model_id}", json.dumps(overrides))
    
    # Log the override
    logger.info(f"Manual override applied: {override_action} for {model_id}")
    
    # Notify via Slack
    from utils.slack import post as send_message
    send_message(
        channel="#data-quality",
        title="🔧 Manual Override Applied",
        details=f"Override '{override_action}' applied for model '{model_id}'.\nExpires in 7 days.",
        urgency="medium"
    )
    
    return {
        "status": "success",
        "model_id": model_id,
        "action": override_action,
        "timestamp": timestamp,
        "expires": (datetime.now() + timedelta(days=7)).isoformat()
    } 