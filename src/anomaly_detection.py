"""
Anomaly Detection Module
Uses Isolation Forest algorithm to detect outliers in AdTech data
"""

import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AnomalyResult:
    """Data class to store anomaly detection results"""
    row_index: int
    date: str
    campaign_id: str
    campaign_name: str
    metric: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # 'high', 'medium', 'low'
    description: str


class AnomalyDetector:
    """
    Detects anomalies in AdTech data using Isolation Forest
    
    Isolation Forest is effective for detecting outliers because:
    - It isolates observations by randomly selecting features
    - Anomalies are easier to isolate and require fewer splits
    - Works well with high-dimensional data
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the anomaly detector
        
        Args:
            contamination: Expected proportion of outliers (default 10%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.anomalies = []
    
    def detect(self, df: pl.DataFrame, 
               feature_cols: Optional[List[str]] = None) -> Tuple[pl.DataFrame, List[AnomalyResult]]:
        """
        Detect anomalies in the dataset
        
        Args:
            df: Input Polars DataFrame
            feature_cols: List of columns to use for anomaly detection
                         If None, uses common AdTech metrics
        
        Returns:
            Tuple of (DataFrame with anomaly scores, List of AnomalyResult)
        """
        # Default feature columns for AdTech data
        if feature_cols is None:
            potential_cols = ['impressions', 'clicks', 'conversions', 'spend', 'revenue', 
                            'ctr', 'conversion_rate', 'cpa', 'roi', 'roas']
            feature_cols = [col for col in potential_cols if col in df.columns]
        
        self.feature_columns = feature_cols
        
        if not feature_cols:
            print("⚠️ No numeric columns found for anomaly detection")
            return df, []
        
        # Convert to numpy for sklearn
        features_df = df.select(feature_cols)
        X = features_df.to_numpy()
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Add predictions to DataFrame
        df = df.with_columns([
            pl.Series('is_anomaly', predictions == -1),
            pl.Series('anomaly_score', anomaly_scores)
        ])
        
        # Extract detailed anomaly information
        self.anomalies = self._extract_anomaly_details(df, feature_cols)
        
        print(f"🔍 Anomaly Detection Complete:")
        print(f"   Total records: {len(df)}")
        print(f"   Anomalies found: {len(self.anomalies)}")
        
        return df, self.anomalies
    
    def _extract_anomaly_details(self, df: pl.DataFrame, 
                                  feature_cols: List[str]) -> List[AnomalyResult]:
        """Extract detailed information about detected anomalies"""
        anomalies = []
        
        # Filter anomalous rows
        anomaly_df = df.filter(pl.col('is_anomaly') == True)
        
        if len(anomaly_df) == 0:
            return anomalies
        
        # Calculate statistics for each feature
        stats = {}
        for col in feature_cols:
            col_data = df[col].to_numpy()
            stats[col] = {
                'mean': float(np.nanmean(col_data)),
                'std': float(np.nanstd(col_data)),
                'min': float(np.nanmin(col_data)),
                'max': float(np.nanmax(col_data))
            }
        
        # Process each anomalous row
        for i, row in enumerate(anomaly_df.iter_rows(named=True)):
            # Find which metric(s) are anomalous
            for col in feature_cols:
                value = row.get(col, 0)
                if value is None:
                    continue
                    
                mean = stats[col]['mean']
                std = stats[col]['std']
                
                # Check if value is outside 2 standard deviations
                if std > 0:
                    z_score = abs(value - mean) / std
                    
                    if z_score > 2:  # Significant deviation
                        severity = 'high' if z_score > 3 else ('medium' if z_score > 2.5 else 'low')
                        
                        # Determine direction
                        direction = "below" if value < mean else "above"
                        pct_diff = ((value - mean) / mean * 100) if mean != 0 else 0
                        
                        # Create description
                        description = self._create_anomaly_description(
                            col, value, mean, pct_diff, direction, row
                        )
                        
                        anomaly = AnomalyResult(
                            row_index=i,
                            date=str(row.get('date', 'Unknown')),
                            campaign_id=str(row.get('campaign_id', 'Unknown')),
                            campaign_name=str(row.get('campaign_name', 'Unknown')),
                            metric=col,
                            value=round(float(value), 2),
                            expected_range=(round(float(mean - 2*std), 2), round(float(mean + 2*std), 2)),
                            severity=severity,
                            description=description
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _create_anomaly_description(self, metric: str, value: float, 
                                     mean: float, pct_diff: float, 
                                     direction: str, row: dict) -> str:
        """Generate human-readable anomaly description"""
        campaign = row.get('campaign_name', 'Unknown Campaign')
        date = row.get('date', 'Unknown Date')
        region = row.get('region', '')
        
        metric_names = {
            'impressions': 'Impressions',
            'clicks': 'Clicks',
            'conversions': 'Conversions',
            'spend': 'Ad Spend',
            'revenue': 'Revenue',
            'ctr': 'Click-Through Rate (CTR)',
            'conversion_rate': 'Conversion Rate',
            'cpa': 'Cost Per Acquisition (CPA)',
            'roi': 'Return on Investment (ROI)',
            'roas': 'Return on Ad Spend (ROAS)'
        }
        
        metric_display = metric_names.get(metric, metric)
        
        if direction == "below":
            desc = f"{metric_display} dropped {abs(pct_diff):.1f}% below average"
        else:
            desc = f"{metric_display} spiked {pct_diff:.1f}% above average"
        
        desc += f" for {campaign} on {date}"
        if region:
            desc += f" in {region}"
        
        return desc
    
    def get_anomaly_summary(self) -> Dict:
        """Get a summary of detected anomalies"""
        if not self.anomalies:
            return {'total': 0, 'by_severity': {}, 'by_metric': {}, 'by_campaign': {}}
        
        summary = {
            'total': len(self.anomalies),
            'by_severity': {'high': 0, 'medium': 0, 'low': 0},
            'by_metric': {},
            'by_campaign': {}
        }
        
        for anomaly in self.anomalies:
            # Count by severity
            summary['by_severity'][anomaly.severity] += 1
            
            # Count by metric
            if anomaly.metric not in summary['by_metric']:
                summary['by_metric'][anomaly.metric] = 0
            summary['by_metric'][anomaly.metric] += 1
            
            # Count by campaign
            if anomaly.campaign_name not in summary['by_campaign']:
                summary['by_campaign'][anomaly.campaign_name] = 0
            summary['by_campaign'][anomaly.campaign_name] += 1
        
        return summary
    
    def to_dict(self) -> List[Dict]:
        """Convert anomalies to list of dictionaries for JSON serialization"""
        return [
            {
                'date': a.date,
                'campaign_id': a.campaign_id,
                'campaign_name': a.campaign_name,
                'metric': a.metric,
                'value': a.value,
                'expected_range': a.expected_range,
                'severity': a.severity,
                'description': a.description
            }
            for a in self.anomalies
        ]


if __name__ == "__main__":
    # Test the module
    from data_ingestion import DataIngestion, DataTransformer
    
    ingestion = DataIngestion()
    transformer = DataTransformer()
    detector = AnomalyDetector(contamination=0.1)
    
    # Load and transform sample data
    sample_path = "data/sample/ad_performance.csv"
    import os
    if os.path.exists(sample_path):
        df = ingestion.load_csv(sample_path)
        df = transformer.parse_dates(df, ['date'])
        df = transformer.calculate_metrics(df)
        
        # Detect anomalies
        df_with_anomalies, anomalies = detector.detect(df)
        
        print("\n🚨 Detected Anomalies:")
        for anomaly in anomalies[:5]:  # Show first 5
            print(f"   - [{anomaly.severity.upper()}] {anomaly.description}")
        
        print("\n📊 Summary:")
        summary = detector.get_anomaly_summary()
        print(f"   Total anomalies: {summary['total']}")
        print(f"   By severity: {summary['by_severity']}")
