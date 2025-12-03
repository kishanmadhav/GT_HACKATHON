"""
Data Normalizer Module
Handles unstructured data - automatically maps columns to expected fields
Works with any CSV, JSON, or SQL data structure
"""

import polars as pl
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class DataNormalizer:
    """
    Automatically normalizes unstructured data into a format the pipeline can use.
    Uses fuzzy matching and type inference to map columns.
    """
    
    # Column mapping patterns - maps various names to standard fields
    COLUMN_PATTERNS = {
        'date': [
            r'date', r'day', r'time', r'timestamp', r'created', r'period',
            r'report_date', r'data_date', r'event_date', r'dt'
        ],
        'campaign_id': [
            r'campaign_id', r'campaign', r'camp_id', r'adgroup_id', r'ad_id',
            r'id', r'campaign_key', r'cid'
        ],
        'campaign_name': [
            r'campaign_name', r'name', r'campaign_title', r'ad_name', r'title',
            r'adgroup_name', r'ad_group', r'campaign$'
        ],
        'impressions': [
            r'impression', r'impr', r'views', r'imp', r'shows', r'reach',
            r'display', r'served'
        ],
        'clicks': [
            r'click', r'clk', r'visits', r'sessions', r'hits'
        ],
        'conversions': [
            r'conversion', r'conv', r'purchase', r'order', r'transaction',
            r'signup', r'lead', r'action', r'goal', r'sale'
        ],
        'spend': [
            r'spend', r'cost', r'expense', r'budget', r'investment',
            r'ad_spend', r'media_cost', r'amount_spent'
        ],
        'revenue': [
            r'revenue', r'income', r'sales', r'value', r'earning',
            r'conversion_value', r'total_value', r'gmv', r'amount'
        ],
        'region': [
            r'region', r'country', r'location', r'geo', r'territory',
            r'market', r'area', r'state', r'city'
        ],
        'platform': [
            r'platform', r'source', r'channel', r'network', r'medium',
            r'publisher', r'ad_platform', r'traffic_source'
        ]
    }
    
    def __init__(self):
        self.column_mapping = {}
        self.unmapped_columns = []
        self.inferred_types = {}
        
    def _match_column(self, col_name: str) -> Optional[str]:
        """Try to match a column name to a standard field"""
        col_lower = col_name.lower().strip()
        
        for standard_field, patterns in self.COLUMN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    return standard_field
        return None
    
    def _infer_column_type(self, df: pl.DataFrame, col: str) -> str:
        """Infer what type of data a column contains"""
        sample = df[col].drop_nulls().head(100)
        
        if len(sample) == 0:
            return 'unknown'
        
        dtype = df[col].dtype
        
        # Check if it's a date
        if dtype == pl.Date or dtype == pl.Datetime:
            return 'date'
        
        # Check string columns for date patterns
        if dtype == pl.Utf8:
            first_val = str(sample[0]) if len(sample) > 0 else ''
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
                r'\d{2}/\d{2}/\d{4}',  # 01/15/2024
                r'\d{2}-\d{2}-\d{4}',  # 15-01-2024
            ]
            for pattern in date_patterns:
                if re.match(pattern, first_val):
                    return 'date'
            
            # Check if it looks like an ID or name
            unique_ratio = sample.n_unique() / len(sample)
            if unique_ratio > 0.5:
                return 'categorical_high_cardinality'  # Likely IDs or names
            return 'categorical'
        
        # Numeric columns
        if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            mean_val = sample.mean()
            max_val = sample.max()
            
            # Large numbers are likely impressions/clicks
            if mean_val and mean_val > 1000:
                return 'metric_large'
            # Decimal numbers with small values might be rates or currency
            elif dtype in [pl.Float64, pl.Float32]:
                return 'metric_decimal'
            else:
                return 'metric_small'
        
        return 'unknown'
    
    def analyze_schema(self, df: pl.DataFrame) -> Dict:
        """Analyze the dataframe and create column mapping"""
        self.column_mapping = {}
        self.unmapped_columns = []
        mapped_standards = set()
        
        # First pass: exact/pattern matching
        for col in df.columns:
            standard_field = self._match_column(col)
            if standard_field and standard_field not in mapped_standards:
                self.column_mapping[col] = standard_field
                mapped_standards.add(standard_field)
            else:
                self.unmapped_columns.append(col)
        
        # Second pass: type inference for unmapped columns
        for col in self.unmapped_columns.copy():
            col_type = self._infer_column_type(df, col)
            self.inferred_types[col] = col_type
            
            # Try to fill gaps based on type
            if col_type == 'date' and 'date' not in mapped_standards:
                self.column_mapping[col] = 'date'
                mapped_standards.add('date')
                self.unmapped_columns.remove(col)
            elif col_type == 'metric_large' and 'impressions' not in mapped_standards:
                self.column_mapping[col] = 'impressions'
                mapped_standards.add('impressions')
                self.unmapped_columns.remove(col)
            elif col_type == 'metric_decimal' and 'revenue' not in mapped_standards:
                self.column_mapping[col] = 'revenue'
                mapped_standards.add('revenue')
                self.unmapped_columns.remove(col)
        
        return {
            'mapping': self.column_mapping,
            'unmapped': self.unmapped_columns,
            'inferred_types': self.inferred_types
        }
    
    def normalize(self, df: pl.DataFrame, custom_mapping: Dict = None) -> Tuple[pl.DataFrame, Dict]:
        """
        Normalize dataframe to standard schema
        
        Args:
            df: Input dataframe with any structure
            custom_mapping: Optional dict to override automatic mapping
            
        Returns:
            Tuple of (normalized dataframe, metadata about transformations)
        """
        # Analyze schema
        analysis = self.analyze_schema(df)
        
        # Apply custom mapping overrides
        if custom_mapping:
            for orig, standard in custom_mapping.items():
                if orig in df.columns:
                    self.column_mapping[orig] = standard
        
        # Rename columns to standard names
        rename_map = {orig: standard for orig, standard in self.column_mapping.items()}
        normalized = df.rename(rename_map)
        
        # Keep track of what we have
        available_fields = set(normalized.columns)
        
        # Create missing required columns with defaults/synthetics
        metadata = {
            'original_columns': df.columns,
            'mapping_applied': rename_map,
            'synthetic_columns': [],
            'available_metrics': []
        }
        
        # Handle date column
        if 'date' not in available_fields:
            # Create synthetic date column
            normalized = normalized.with_columns(
                pl.lit(datetime.now().strftime('%Y-%m-%d')).alias('date')
            )
            metadata['synthetic_columns'].append('date')
        else:
            # Try to parse date if it's a string
            if normalized['date'].dtype == pl.Utf8:
                try:
                    normalized = normalized.with_columns(
                        pl.col('date').str.to_date().alias('date')
                    )
                except:
                    # Keep as string if parsing fails
                    pass
        
        # Handle campaign_name
        if 'campaign_name' not in available_fields:
            if 'campaign_id' in available_fields:
                normalized = normalized.with_columns(
                    pl.col('campaign_id').cast(pl.Utf8).alias('campaign_name')
                )
            else:
                normalized = normalized.with_columns(
                    pl.lit('Campaign').alias('campaign_name')
                )
            metadata['synthetic_columns'].append('campaign_name')
        
        # Handle campaign_id
        if 'campaign_id' not in available_fields:
            if 'campaign_name' in available_fields:
                normalized = normalized.with_columns(
                    pl.col('campaign_name').alias('campaign_id')
                )
            else:
                normalized = normalized.with_columns(
                    pl.arange(0, len(normalized)).alias('campaign_id')
                )
            metadata['synthetic_columns'].append('campaign_id')
        
        # Handle numeric metrics - create defaults if missing
        numeric_defaults = {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'spend': 0.0,
            'revenue': 0.0
        }
        
        for field, default in numeric_defaults.items():
            if field in available_fields or field in normalized.columns:
                metadata['available_metrics'].append(field)
            else:
                # Try to find any numeric column we haven't used
                unused_numeric = None
                for col in self.unmapped_columns:
                    if col in normalized.columns:
                        dtype = normalized[col].dtype
                        if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                            unused_numeric = col
                            break
                
                if unused_numeric:
                    normalized = normalized.rename({unused_numeric: field})
                    self.unmapped_columns.remove(unused_numeric)
                    metadata['available_metrics'].append(field)
                else:
                    normalized = normalized.with_columns(
                        pl.lit(default).alias(field)
                    )
                    metadata['synthetic_columns'].append(field)
        
        # Handle region and platform
        if 'region' not in normalized.columns:
            normalized = normalized.with_columns(pl.lit('Unknown').alias('region'))
            metadata['synthetic_columns'].append('region')
        
        if 'platform' not in normalized.columns:
            normalized = normalized.with_columns(pl.lit('Unknown').alias('platform'))
            metadata['synthetic_columns'].append('platform')
        
        return normalized, metadata


def load_and_normalize(file_path: str, custom_mapping: Dict = None) -> Tuple[pl.DataFrame, Dict]:
    """
    Load any CSV/JSON/Parquet file and normalize it
    
    Args:
        file_path: Path to data file
        custom_mapping: Optional column mapping overrides
        
    Returns:
        Tuple of (normalized dataframe, metadata)
    """
    # Detect file type and load
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'csv':
        df = pl.read_csv(file_path, try_parse_dates=True)
    elif ext == 'json':
        # Handle both JSON array and JSON lines
        try:
            df = pl.read_json(file_path)
        except:
            df = pl.read_ndjson(file_path)
    elif ext == 'parquet':
        df = pl.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Normalize
    normalizer = DataNormalizer()
    return normalizer.normalize(df, custom_mapping)


if __name__ == "__main__":
    # Test with sample data
    import os
    
    test_path = "data/sample/ad_performance.csv"
    if os.path.exists(test_path):
        df, metadata = load_and_normalize(test_path)
        print("Normalized DataFrame:")
        print(df.head())
        print("\nMetadata:")
        print(json.dumps(metadata, indent=2, default=str))
