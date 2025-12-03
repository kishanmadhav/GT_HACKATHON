"""
Data Ingestion Module
Handles loading data from various sources (CSV, SQL, JSON)
Uses Polars for fast, memory-efficient data processing
"""

import polars as pl
import os
from pathlib import Path
from typing import Union, Optional
import json


class DataIngestion:
    """Handles data ingestion from multiple sources"""
    
    SUPPORTED_FORMATS = ['.csv', '.json', '.parquet']
    
    def __init__(self):
        self.data = None
        self.metadata = {}
    
    def load_csv(self, file_path: str, **kwargs) -> pl.DataFrame:
        """
        Load data from CSV file using Polars
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pl.read_csv
            
        Returns:
            Polars DataFrame
        """
        try:
            df = pl.read_csv(file_path, **kwargs)
            self.data = df
            self.metadata = {
                'source': file_path,
                'format': 'csv',
                'rows': len(df),
                'columns': df.columns,
                'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
            }
            print(f"✅ Loaded CSV: {file_path}")
            print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
    
    def load_json(self, file_path: str) -> pl.DataFrame:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Polars DataFrame
        """
        try:
            df = pl.read_json(file_path)
            self.data = df
            self.metadata = {
                'source': file_path,
                'format': 'json',
                'rows': len(df),
                'columns': df.columns
            }
            print(f"✅ Loaded JSON: {file_path}")
            return df
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            raise
    
    def load_parquet(self, file_path: str) -> pl.DataFrame:
        """
        Load data from Parquet file
        
        Args:
            file_path: Path to Parquet file
            
        Returns:
            Polars DataFrame
        """
        try:
            df = pl.read_parquet(file_path)
            self.data = df
            self.metadata = {
                'source': file_path,
                'format': 'parquet',
                'rows': len(df),
                'columns': df.columns
            }
            print(f"✅ Loaded Parquet: {file_path}")
            return df
        except Exception as e:
            print(f"❌ Error loading Parquet: {e}")
            raise
    
    def load_auto(self, file_path: str) -> pl.DataFrame:
        """
        Automatically detect file format and load data
        
        Args:
            file_path: Path to data file
            
        Returns:
            Polars DataFrame
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == '.csv':
            return self.load_csv(file_path)
        elif ext == '.json':
            return self.load_json(file_path)
        elif ext == '.parquet':
            return self.load_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {self.SUPPORTED_FORMATS}")
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of loaded data
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            return {}
        
        df = self.data
        summary = {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'columns': df.columns,
            'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            'null_counts': {col: df[col].null_count() for col in df.columns},
            'numeric_stats': {}
        }
        
        # Get statistics for numeric columns
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                       if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
        
        for col in numeric_cols:
            summary['numeric_stats'][col] = {
                'mean': float(df[col].mean()) if df[col].mean() is not None else None,
                'min': float(df[col].min()) if df[col].min() is not None else None,
                'max': float(df[col].max()) if df[col].max() is not None else None,
                'std': float(df[col].std()) if df[col].std() is not None else None
            }
        
        return summary


class DataTransformer:
    """Handles data transformation and preprocessing"""
    
    @staticmethod
    def parse_dates(df: pl.DataFrame, date_columns: list) -> pl.DataFrame:
        """Convert string columns to date type"""
        for col in date_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).str.to_date().alias(col)
                )
        return df
    
    @staticmethod
    def calculate_metrics(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate common AdTech metrics
        
        Args:
            df: Input DataFrame with impressions, clicks, conversions, spend, revenue
            
        Returns:
            DataFrame with additional metric columns
        """
        # Check required columns exist
        required = ['impressions', 'clicks', 'conversions', 'spend', 'revenue']
        available = [col for col in required if col in df.columns]
        
        if 'impressions' in df.columns and 'clicks' in df.columns:
            df = df.with_columns(
                (pl.col('clicks') / pl.col('impressions') * 100).round(2).alias('ctr')
            )
        
        if 'conversions' in df.columns and 'clicks' in df.columns:
            df = df.with_columns(
                (pl.col('conversions') / pl.col('clicks') * 100).round(2).alias('conversion_rate')
            )
        
        if 'spend' in df.columns and 'conversions' in df.columns:
            df = df.with_columns(
                (pl.col('spend') / pl.col('conversions')).round(2).alias('cpa')
            )
        
        if 'revenue' in df.columns and 'spend' in df.columns:
            df = df.with_columns(
                ((pl.col('revenue') - pl.col('spend')) / pl.col('spend') * 100).round(2).alias('roi')
            )
            df = df.with_columns(
                (pl.col('revenue') / pl.col('spend')).round(2).alias('roas')
            )
        
        return df
    
    @staticmethod
    def aggregate_by_date(df: pl.DataFrame, date_col: str = 'date') -> pl.DataFrame:
        """Aggregate metrics by date"""
        numeric_cols = ['impressions', 'clicks', 'conversions', 'spend', 'revenue']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        agg_exprs = [pl.col(col).sum().alias(col) for col in available_cols]
        
        return df.group_by(date_col).agg(agg_exprs).sort(date_col)
    
    @staticmethod
    def aggregate_by_campaign(df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate metrics by campaign"""
        group_cols = ['campaign_id', 'campaign_name']
        available_group = [col for col in group_cols if col in df.columns]
        
        if not available_group:
            return df
        
        numeric_cols = ['impressions', 'clicks', 'conversions', 'spend', 'revenue']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        agg_exprs = [pl.col(col).sum().alias(col) for col in available_cols]
        
        return df.group_by(available_group).agg(agg_exprs)


if __name__ == "__main__":
    # Test the module
    ingestion = DataIngestion()
    transformer = DataTransformer()
    
    # Load sample data
    sample_path = "data/sample/ad_performance.csv"
    if os.path.exists(sample_path):
        df = ingestion.load_csv(sample_path)
        print("\n📊 Data Summary:")
        print(df.head())
        
        # Transform data
        df = transformer.parse_dates(df, ['date'])
        df = transformer.calculate_metrics(df)
        print("\n📈 With Calculated Metrics:")
        print(df.head())
