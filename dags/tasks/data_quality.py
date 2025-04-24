import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
import mlflow
import gc
from utils.cache import GLOBAL_CACHE, cache_result
import os
import warnings

# Suppress warnings during data quality checks
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """Monitor for data quality checks and drift detection.
    Optimized for memory efficiency with large datasets."""
    
    def __init__(self, 
                 reference_data_path=None, 
                 missing_value_threshold=0.05, 
                 outlier_threshold=3.0,
                 drift_threshold=0.1,
                 correlation_threshold=0.3,
                 max_columns_analyzed=30,
                 sample_size=10000,
                 chunk_size=100000,
                 memory_limit_mb=8000):
        """
        Initialize the DataQualityMonitor with configurable thresholds.
        
        Args:
            reference_data_path: Path to reference data for drift detection
            missing_value_threshold: Threshold for flagging columns with missing values
            outlier_threshold: Z-score threshold for outlier detection
            drift_threshold: Threshold for detecting data drift
            correlation_threshold: Threshold for detecting correlation changes
            max_columns_analyzed: Maximum number of columns to analyze for memory efficiency
            sample_size: Sample size for large datasets to use during statistical tests
            chunk_size: Process data in chunks of this size for extreme memory efficiency
            memory_limit_mb: Memory limit in MB to trigger more aggressive optimizations
        """
        self.reference_data_path = reference_data_path
        self.missing_value_threshold = missing_value_threshold
        self.outlier_threshold = outlier_threshold
        self.drift_threshold = drift_threshold
        self.correlation_threshold = correlation_threshold
        self.max_columns_analyzed = max_columns_analyzed
        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.reference_data = None
        self.logger = logging.getLogger(__name__)
        
        # Add memory tracking
        self.enable_memory_tracking = True
        
        # Load reference data if provided
        if reference_data_path and os.path.exists(reference_data_path):
            try:
                self._load_reference_data()
            except Exception as e:
                self.logger.error(f"Failed to load reference data: {str(e)}")
    
    def _log_memory_usage(self, step_name):
        """Log memory usage at a specific step if enabled"""
        if not self.enable_memory_tracking:
            return
            
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.logger.info(f"Memory usage at {step_name}: {memory_mb:.2f} MB")
            
            # Trigger aggressive GC if memory usage is too high
            if memory_mb > self.memory_limit_mb:
                self.logger.warning(f"Memory usage too high ({memory_mb:.2f} MB). Triggering aggressive garbage collection.")
                self._reduce_memory_footprint()
        except Exception as e:
            self.logger.warning(f"Could not log memory usage: {str(e)}")
    
    def _reduce_memory_footprint(self):
        """Aggressively reduce memory footprint"""
        try:
            # Force full garbage collection
            gc.collect(2)
            
            # Clear any cached objects in global cache if available
            if 'GLOBAL_CACHE' in globals():
                try:
                    for key in list(GLOBAL_CACHE.keys()):
                        if 'temp_' in key or 'cache_' in key:
                            GLOBAL_CACHE.pop(key, None)
                except:
                    pass
            
            # Try to release memory back to OS on Linux
            if hasattr(gc, 'malloc_trim'):
                gc.malloc_trim(0)
        except Exception as e:
            self.logger.warning(f"Error during memory reduction: {str(e)}")
            
    def _load_reference_data(self):
        """Load reference data for drift detection with memory optimization"""
        if not self.reference_data_path:
            return None
            
        try:
            # Check if file exists
            if not os.path.exists(self.reference_data_path):
                self.logger.warning(f"Reference data file not found: {self.reference_data_path}")
                return None
                
            # Use pyarrow to get metadata before loading
            import pyarrow.parquet as pq
            metadata = pq.read_metadata(self.reference_data_path)
            num_rows = metadata.num_rows
            
            # For large reference datasets, use sampling
            if num_rows > self.sample_size:
                self.logger.info(f"Reference dataset has {num_rows} rows, sampling to {self.sample_size}")
                
                # Use efficient pyarrow-based sampling
                import pyarrow as pa
                
                # Read the file
                table = pq.read_table(self.reference_data_path)
                
                # Sample down to manageable size
                import numpy as np
                indices = sorted(np.random.choice(num_rows, size=self.sample_size, replace=False))
                sampled_table = table.take(pa.array(indices))
                
                # Convert to pandas
                self.reference_data = sampled_table.to_pandas()
                
                # Clean up
                del table
                del sampled_table
                gc.collect()
            else:
                # For small files, load directly
                self.reference_data = pd.read_parquet(self.reference_data_path)
                
            self.logger.info(f"Loaded reference data with shape {self.reference_data.shape}")
            
            # Optimize dtypes to reduce memory usage
            self.reference_data = self._optimize_dtypes(self.reference_data)
            
            return self.reference_data
            
        except Exception as e:
            self.logger.error(f"Error loading reference data: {str(e)}")
            return None
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        try:
            # Process numeric columns
            for col in df.select_dtypes(include=['int']).columns:
                # Downcast integers
                if df[col].min() >= 0:  # Unsigned int
                    if df[col].max() <= 255:
                        df[col] = df[col].astype(np.uint8)
                    elif df[col].max() <= 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif df[col].max() <= 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:  # Signed int
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype(np.int8)
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype(np.int16)
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            
            # Downcast floats
            for col in df.select_dtypes(include=['float']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
                
            # Convert object columns to categorical if cardinality is low
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
                    
            return df
        except Exception as e:
            self.logger.warning(f"Error optimizing dtypes: {str(e)}")
            return df
            
    def _select_columns_for_analysis(self, df, analysis_type="numerical"):
        """
        Intelligently select columns for analysis, limiting to most relevant ones
        for memory efficiency.
        
        Args:
            df: DataFrame to analyze
            analysis_type: Type of analysis. Can be "numerical", "categorical", or "all"
            
        Returns:
            List of column names to analyze
        """
        # First filter by type
        if analysis_type == "numerical":
            candidate_cols = df.select_dtypes(include=['number']).columns.tolist()
        elif analysis_type == "categorical":
            candidate_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:  # "all"
            candidate_cols = df.columns.tolist()
            
        # If number of columns exceeds max, prioritize
        if len(candidate_cols) > self.max_columns_analyzed:
            self.logger.info(f"Limiting analysis to {self.max_columns_analyzed} columns")
            
            if analysis_type == "numerical":
                # For numerical, prioritize by variance and missing values
                try:
                    # Calculate variance-to-mean ratio (VMR) to find most variable columns
                    vmr = {}
                    for col in candidate_cols:
                        mean = df[col].mean()
                        if mean != 0:  # Avoid division by zero
                            vmr[col] = df[col].var() / mean if not np.isnan(mean) else 0
                        else:
                            vmr[col] = df[col].var()  # Just use variance if mean is zero
                            
                    # Sort columns by VMR and select top N
                    sorted_cols = sorted(vmr.items(), key=lambda x: x[1], reverse=True)
                    selected_cols = [col for col, _ in sorted_cols[:self.max_columns_analyzed]]
                    
                    # Make sure we include key columns by name patterns
                    key_patterns = ['id', 'target', 'label', 'score', 'probability', 'prediction']
                    for col in candidate_cols:
                        if any(pattern in col.lower() for pattern in key_patterns) and col not in selected_cols:
                            # Replace the least variable column
                            if len(selected_cols) >= self.max_columns_analyzed:
                                selected_cols.pop()
                            selected_cols.append(col)
                            
                    return selected_cols
                    
                except Exception as e:
                    self.logger.warning(f"Error during column selection: {str(e)}")
                    # Fall back to simple selection
            
            # Simple selection method as fallback
            # Include ID and target columns first
            selected_cols = []
            
            # Add suspected ID columns
            id_cols = [col for col in candidate_cols if 'id' in col.lower()]
            selected_cols.extend(id_cols[:min(3, len(id_cols))])  # Max 3 ID columns
            
            # Add suspected target columns
            target_cols = [col for col in candidate_cols if any(name in col.lower() for name in 
                          ['target', 'label', 'class', 'prediction', 'score', 'probability'])]
            selected_cols.extend(target_cols[:min(5, len(target_cols))])  # Max 5 target columns
            
            # Add remaining columns up to the limit
            remaining_slots = self.max_columns_analyzed - len(selected_cols)
            if remaining_slots > 0:
                other_cols = [col for col in candidate_cols if col not in selected_cols]
                selected_cols.extend(other_cols[:remaining_slots])
                
            return selected_cols
            
        # If under the limit, return all candidates
        return candidate_cols
     
    def _process_in_chunks(self, df: pd.DataFrame, func, chunk_size: Optional[int] = None) -> Dict:
        """
        Process a DataFrame in chunks to save memory
        
        Args:
            df: DataFrame to process
            func: Function to apply to each chunk, should accept DataFrame and return dict
            chunk_size: Size of chunks to process
            
        Returns:
            Combined results from all chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        # If DataFrame is small enough, process directly
        if len(df) <= chunk_size:
            return func(df)
            
        # Otherwise process in chunks
        self.logger.info(f"Processing DataFrame with {len(df)} rows in chunks of {chunk_size}")
        
        results = {}
        # Process DataFrame in chunks
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_results = func(chunk)
            
            # Merge chunk results with overall results
            for key, value in chunk_results.items():
                if key not in results:
                    results[key] = value
                elif isinstance(value, dict):
                    results[key].update(value)
                elif isinstance(value, (int, float)):
                    results[key] = (results[key] + value) / 2  # Average numeric values
                    
            # Clean up
            del chunk
            gc.collect()
            
        return results   
        
    def check_missing_values(self, df):
        """
        Check for missing values in each column.
        Memory-optimized for large datasets.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary of columns with missing values above threshold
        """
        self._log_memory_usage("before_missing_values_check")
        
        # Define a function to process a chunk
        def process_chunk(chunk_df):
            # Select columns for analysis - check all columns for missing values
            columns_to_check = chunk_df.columns.tolist()
            
            # For very large DataFrames, process in smaller batches to save memory
            missing_values = {}
            batch_size = min(50, max(1, len(columns_to_check) // 10))  # Process 10 batches, or 50 columns at once max
            
            for i in range(0, len(columns_to_check), batch_size):
                batch_columns = columns_to_check[i:i+batch_size]
                
                # Process this batch
                for column in batch_columns:
                    # Calculate missing value percentage efficiently using isnull().mean()
                    # which is memory-efficient for large datasets
                    missing_pct = chunk_df[column].isnull().mean()
                    
                    if missing_pct > self.missing_value_threshold:
                        missing_values[column] = float(missing_pct)  # Convert to float for JSON serialization
                
                # Explicit garbage collection
                gc.collect()
                
            return {'missing_values': missing_values}
            
        # Process in chunks if needed
        if len(df) > self.chunk_size:
            # For missing values, we need accurate counts across the entire dataset
            # So we'll use our chunking with a different approach
            missing_values = {}
            
            # Calculate missing count for each column across entire dataset
            for col in df.columns:
                # Count non-null values in chunks to save memory
                non_null_count = 0
                total_count = 0
                
                for i in range(0, len(df), self.chunk_size):
                    chunk = df.iloc[i:i+self.chunk_size]
                    non_null_count += chunk[col].count()
                    total_count += len(chunk)
                    del chunk
                
                # Calculate missing percentage
                if total_count > 0:
                    missing_pct = 1.0 - (non_null_count / total_count)
                    if missing_pct > self.missing_value_threshold:
                        missing_values[col] = float(missing_pct)
                
                # Clean up after each column
                gc.collect()
        else:
            result = process_chunk(df)
            missing_values = result['missing_values']
        
        self._log_memory_usage("after_missing_values_check")
        return missing_values
        
    def check_outliers(self, df):
        """
        Check for outliers in numerical columns using z-score method.
        Memory-optimized for large datasets.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary of columns with outlier counts
        """
        self._log_memory_usage("before_outlier_check")
        
        # Define function to process a subset of data
        def process_outliers(subset_df):
            # Select only numerical columns for outlier detection
            num_columns = self._select_columns_for_analysis(subset_df, "numerical")
            outliers = {}
            
            for column in num_columns:
                try:
                    # Skip columns with too many missing values
                    missing_rate = subset_df[column].isnull().mean()
                    if missing_rate > 0.5:  # Skip if more than 50% missing
                        continue
                        
                    # Get only finite values for z-score calculation
                    series = subset_df[column].dropna()
                    
                    if len(series) < 10:  # Need sufficient data
                        continue
                        
                    # For very large series, sample to calculate stats
                    if len(series) > 10000:
                        series = series.sample(10000, random_state=42)
                        
                    # Calculate z-scores efficiently by chunks if needed
                    mean = series.mean()
                    std = series.std()
                    
                    if std == 0:  # Avoid division by zero
                        continue
                    
                    # Calculate z-scores without creating a full series
                    # This is more memory efficient for large datasets
                    outlier_count = 0
                    # Process in small chunks of 1000 values
                    for i in range(0, len(series), 1000):
                        chunk = series.iloc[i:i+1000]
                        z_scores = np.abs((chunk - mean) / std)
                        outlier_count += (z_scores > self.outlier_threshold).sum()
                        del z_scores
                        
                    outlier_percent = outlier_count / len(subset_df) * 100
                    
                    # Only include if we found substantial outliers
                    if outlier_count > 10 or outlier_percent > 1.0:
                        outliers[column] = {
                            'count': int(outlier_count),
                            'percent': float(outlier_percent)
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Error detecting outliers in column {column}: {str(e)}")
                    
                # Clean up after each column
                gc.collect()
                
            return {'outliers': outliers}
            
        # For very large datasets, sample down before outlier detection
        df_sample = df
        if len(df) > self.sample_size:
            self.logger.info(f"Sampling {self.sample_size} rows for outlier detection")
            df_sample = df.sample(min(self.sample_size, len(df)), random_state=42)
            
        # Process the sample
        result = process_outliers(df_sample)
        outliers = result['outliers']
        
        # Clean up
        if df_sample is not df:
            del df_sample
            gc.collect()
        
        self._log_memory_usage("after_outlier_check")
        return outliers
    
    def check_data_drift(self, df):
        """
        Check for data drift compared to reference data.
        Memory-optimized for large datasets.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary of columns with detected drift
        """
        self._log_memory_usage("before_drift_check")
        
        # If no reference data, we can't check for drift
        if self.reference_data is None:
            if self.reference_data_path:
                self.logger.warning("No reference data available for drift detection")
            return {}
            
        # Get common columns
        common_columns = list(set(df.columns) & set(self.reference_data.columns))
        
        # Select columns to analyze - prioritize numerical for statistical tests
        numerical_columns = self._select_columns_for_analysis(
            df[common_columns], 
            "numerical"
        )
        
        # For very large datasets, use sampling for both current and reference
        current_sample = df
        reference_sample = self.reference_data
        
        # Sample with a two-phase approach for very large datasets
        if len(df) > self.sample_size * 10:  # For extremely large datasets
            # First sample to intermediate size
            intermediate_size = min(len(df) // 10, self.sample_size * 2)
            current_sample = df.sample(intermediate_size, random_state=42)
            # Then sample to final size
            current_sample = current_sample.sample(min(self.sample_size, len(current_sample)), random_state=42)
        elif len(df) > self.sample_size:
            current_sample = df.sample(min(self.sample_size, len(df)), random_state=42)
            
        if len(self.reference_data) > self.sample_size:
            reference_sample = self.reference_data.sample(min(self.sample_size, len(self.reference_data)), random_state=42)
            
        # Check for drift in each column
        drift_detected = {}
        
        # Process columns in batches to save memory
        batch_size = min(10, len(numerical_columns))
        for i in range(0, len(numerical_columns), batch_size):
            batch_columns = numerical_columns[i:i+batch_size]
            
            for column in batch_columns:
                try:
                    # Get data from both datasets, dropping nulls
                    current_data = current_sample[column].dropna()
                    reference_data = reference_sample[column].dropna()
                    
                    # Skip if we don't have enough data
                    if len(current_data) < 30 or len(reference_data) < 30:
                        continue
                    
                    # For very large series, sample down to a reasonable size for statistical tests
                    max_size_for_test = 5000  # Reasonable size for KS test
                    if len(current_data) > max_size_for_test:
                        current_data = current_data.sample(max_size_for_test, random_state=42)
                    if len(reference_data) > max_size_for_test:
                        reference_data = reference_data.sample(max_size_for_test, random_state=42)
                        
                    # Use KS test for detecting distribution differences
                    from scipy import stats
                    ks_statistic, p_value = stats.ks_2samp(current_data, reference_data)
                    
                    # Check if drift detected based on p-value
                    if p_value < 0.05 and ks_statistic > self.drift_threshold:
                        # Calculate additional statistics for the report
                        current_mean = float(current_data.mean())
                        reference_mean = float(reference_data.mean())
                        mean_diff_pct = abs((current_mean - reference_mean) / reference_mean) * 100 if reference_mean != 0 else 0
                        
                        drift_detected[column] = {
                            'ks_statistic': float(ks_statistic),
                            'p_value': float(p_value),
                            'current_mean': current_mean,
                            'reference_mean': reference_mean,
                            'mean_diff_pct': float(mean_diff_pct)
                        }
                        
                    # Clean up
                    del current_data
                    del reference_data
                    
                except Exception as e:
                    self.logger.warning(f"Error detecting drift for column {column}: {str(e)}")
            
            # Clean up after each batch
            gc.collect()
                
        self._log_memory_usage("after_drift_check")
        return drift_detected
        
    def check_correlation_changes(self, df):
        """
        Check for correlation changes compared to reference data.
        Memory-optimized implementation for large datasets.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary of column pairs with correlation changes
        """
        self._log_memory_usage("before_correlation_check")
        
        # If no reference data, we can't check for correlation changes
        if self.reference_data is None:
            return {}
            
        # Get common columns
        common_columns = list(set(df.columns) & set(self.reference_data.columns))
        
        # Select numerical columns for correlation analysis
        # Use fewer columns for correlation to limit memory use (correlations are quadratic)
        numerical_columns = self._select_columns_for_analysis(
            df[common_columns], 
            "numerical"
        )[:min(15, self.max_columns_analyzed)]  # Further reduce to 15 for correlation analysis
        
        if len(numerical_columns) < 2:
            self.logger.info("Not enough numerical columns for correlation analysis")
            return {}
            
        # Sample data if needed
        current_sample = df[numerical_columns]
        reference_sample = self.reference_data[numerical_columns]
        
        # More aggressive sampling for correlation analysis
        max_sample = min(5000, self.sample_size)  # Smaller sample for correlation
        
        if len(df) > max_sample:
            current_sample = df[numerical_columns].sample(max_sample, random_state=42)
            
        if len(self.reference_data) > max_sample:
            reference_sample = self.reference_data[numerical_columns].sample(max_sample, random_state=42)
            
        # Calculate correlations - memory efficient implementation
        correlation_changes = {}
        
        try:
            # For extreme memory efficiency, calculate correlation pairwise instead of full matrix
            # This approach uses much less memory for large column sets
            if len(numerical_columns) > 10:  # Only use this approach for many columns
                self.logger.info("Using pairwise correlation calculation to save memory")
                
                for i, col1 in enumerate(numerical_columns):
                    # Only process every other pair to reduce computation
                    for col2 in numerical_columns[i+1:]:
                        try:
                            # Get data for just these two columns
                            curr_df = current_sample[[col1, col2]].dropna()
                            ref_df = reference_sample[[col1, col2]].dropna()
                            
                            # Skip if insufficient data
                            if len(curr_df) < 30 or len(ref_df) < 30:
                                continue
                                
                            # Calculate correlation for this pair
                            curr_corr_val = curr_df[col1].corr(curr_df[col2])
                            ref_corr_val = ref_df[col1].corr(ref_df[col2])
                            
                            # Skip if NaN
                            if pd.isna(curr_corr_val) or pd.isna(ref_corr_val):
                                continue
                                
                            # Check if correlation change is significant
                            corr_diff = abs(curr_corr_val - ref_corr_val)
                            
                            if corr_diff > self.correlation_threshold:
                                correlation_changes[f"{col1}~{col2}"] = {
                                    'current': float(curr_corr_val),
                                    'reference': float(ref_corr_val),
                                    'difference': float(corr_diff)
                                }
                                
                            # Clean up for this pair
                            del curr_df
                            del ref_df
                            gc.collect()
                            
                        except Exception as e:
                            self.logger.warning(f"Error comparing correlation for {col1}-{col2}: {str(e)}")
            else:
                # For smaller column sets, use the full correlation matrix approach
                # Calculate current correlations
                current_corr = current_sample.corr(method='pearson', numeric_only=True)
                
                # Calculate reference correlations
                reference_corr = reference_sample.corr(method='pearson', numeric_only=True)
                
                # Find correlation changes
                for i, col1 in enumerate(numerical_columns):
                    # Only compare with columns after this one to avoid duplicates
                    for col2 in numerical_columns[i+1:]:
                        try:
                            # Get correlations
                            curr_corr_val = current_corr.loc[col1, col2]
                            ref_corr_val = reference_corr.loc[col1, col2]
                            
                            # Skip if NaN
                            if pd.isna(curr_corr_val) or pd.isna(ref_corr_val):
                                continue
                                
                            # Check if correlation change is significant
                            corr_diff = abs(curr_corr_val - ref_corr_val)
                            
                            if corr_diff > self.correlation_threshold:
                                correlation_changes[f"{col1}~{col2}"] = {
                                    'current': float(curr_corr_val),
                                    'reference': float(ref_corr_val),
                                    'difference': float(corr_diff)
                                }
                        except Exception as e:
                            self.logger.warning(f"Error comparing correlation for {col1}-{col2}: {str(e)}")
                            
                # Cleanup
                del current_corr
                del reference_corr
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {str(e)}")
            
        # Final cleanup
        del current_sample
        del reference_sample
        gc.collect()
            
        self._log_memory_usage("after_correlation_check")
        return correlation_changes
        
    def run_quality_checks(self, df):
        """
        Run all data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality check results
        """
        self.logger.info(f"Running data quality checks on DataFrame with shape {df.shape}")
        
        # Optimize data types first to reduce memory usage
        try:
            df = self._optimize_dtypes(df)
            self.logger.info("Optimized DataFrame data types for memory efficiency")
        except Exception as e:
            self.logger.warning(f"Error optimizing data types: {str(e)}")
        
        # Create timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Initialize results
        quality_results = {
            'timestamp': timestamp,
            'data_shape': {
                'rows': len(df),
                'columns': len(df.columns)
            }
        }
        
        # Run checks with memory tracking
        self._log_memory_usage("start_quality_checks")
        
        # Check for missing values
        try:
            self.logger.info("Checking for missing values")
            missing_values = self.check_missing_values(df)
            quality_results['missing_values'] = missing_values
            quality_results['missing_value_issues'] = len(missing_values)
            self.logger.info(f"Found {len(missing_values)} columns with missing values above threshold")
        except Exception as e:
            self.logger.error(f"Error checking missing values: {str(e)}")
            quality_results['missing_values'] = {}
            quality_results['missing_value_issues'] = 0
            
        # Memory cleanup
        gc.collect()
        
        # Check for outliers
        try:
            self.logger.info("Checking for outliers")
            outliers = self.check_outliers(df)
            quality_results['outliers'] = outliers
            quality_results['outlier_issues'] = len(outliers)
            self.logger.info(f"Found {len(outliers)} columns with outliers")
        except Exception as e:
            self.logger.error(f"Error checking outliers: {str(e)}")
            quality_results['outliers'] = {}
            quality_results['outlier_issues'] = 0
            
        # Memory cleanup
        gc.collect()
        
        # Check for data drift if reference data is available
        if self.reference_data is not None:
            try:
                self.logger.info("Checking for data drift")
                data_drift = self.check_data_drift(df)
                quality_results['data_drift'] = data_drift
                self.logger.info(f"Found data drift in {len(data_drift)} columns")
            except Exception as e:
                self.logger.error(f"Error checking data drift: {str(e)}")
                quality_results['data_drift'] = {}
                
            # Memory cleanup
            gc.collect()
            
            # Check for correlation changes
            try:
                self.logger.info("Checking for correlation changes")
                correlation_changes = self.check_correlation_changes(df)
                quality_results['correlation_changes'] = correlation_changes
                self.logger.info(f"Found correlation changes in {len(correlation_changes)} column pairs")
            except Exception as e:
                self.logger.error(f"Error checking correlation changes: {str(e)}")
                quality_results['correlation_changes'] = {}
                
            # Memory cleanup
            gc.collect()
        else:
            quality_results['data_drift'] = {}
            quality_results['correlation_changes'] = {}
            
        # Calculate total issues
        quality_results['total_issues'] = (
            quality_results.get('missing_value_issues', 0) +
            quality_results.get('outlier_issues', 0) +
            len(quality_results.get('data_drift', {})) +
            len(quality_results.get('correlation_changes', {}))
        )
        
        self._log_memory_usage("end_quality_checks")
        
        return quality_results


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