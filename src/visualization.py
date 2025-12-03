"""
Visualization Module
Creates charts and graphs using Plotly for the PDF report
"""

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import os
from datetime import datetime


class ChartGenerator:
    """
    Generates various charts for AdTech reports using Plotly
    """
    
    # Professional color palette
    COLORS = {
        'primary': '#2E86AB',      # Blue
        'secondary': '#A23B72',     # Purple
        'success': '#28A745',       # Green
        'warning': '#FFC107',       # Yellow
        'danger': '#DC3545',        # Red
        'info': '#17A2B8',          # Cyan
        'light': '#F8F9FA',         # Light gray
        'dark': '#343A40'           # Dark gray
    }
    
    COLOR_PALETTE = ['#2E86AB', '#A23B72', '#28A745', '#FFC107', '#DC3545', 
                     '#17A2B8', '#6F42C1', '#FD7E14', '#20C997', '#E83E8C']
    
    def __init__(self, output_dir: str = 'output/charts'):
        """
        Initialize the chart generator
        
        Args:
            output_dir: Directory to save chart images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_time_series_chart(self, df: pl.DataFrame, 
                                  metric: str = 'impressions',
                                  title: Optional[str] = None) -> str:
        """
        Create a time series chart for a specific metric
        
        Args:
            df: DataFrame with date and metric columns
            metric: Column name for the metric
            title: Chart title
            
        Returns:
            Path to saved chart image
        """
        # Aggregate by date
        agg_df = df.group_by('date').agg(pl.col(metric).sum())
        agg_df = agg_df.sort('date')
        
        # Convert to pandas for Plotly
        pdf = agg_df.to_pandas()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pdf['date'],
            y=pdf[metric],
            mode='lines+markers',
            line=dict(color=self.COLORS['primary'], width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.2)',
            name=metric.replace('_', ' ').title()
        ))
        
        fig.update_layout(
            title=dict(
                text=title or f'{metric.replace("_", " ").title()} Over Time',
                font=dict(size=18, color=self.COLORS['dark'])
            ),
            xaxis_title='Date',
            yaxis_title=metric.replace('_', ' ').title(),
            template='plotly_white',
            height=400,
            margin=dict(t=50, l=60, r=30, b=50)
        )
        
        # Save chart
        filename = f'{self.output_dir}/timeseries_{metric}.png'
        fig.write_image(filename, scale=2)
        
        return filename
    
    def create_campaign_comparison_chart(self, df: pl.DataFrame, 
                                          metric: str = 'revenue') -> str:
        """
        Create a bar chart comparing campaigns by a metric
        
        Args:
            df: DataFrame with campaign data
            metric: Metric to compare
            
        Returns:
            Path to saved chart image
        """
        # Aggregate by campaign
        group_cols = ['campaign_name'] if 'campaign_name' in df.columns else ['campaign_id']
        agg_df = df.group_by(group_cols[0]).agg(pl.col(metric).sum()).sort(metric, descending=True)
        
        pdf = agg_df.to_pandas()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=pdf[group_cols[0]],
            y=pdf[metric],
            marker_color=self.COLOR_PALETTE[:len(pdf)],
            text=pdf[metric].apply(lambda x: f'${x:,.0f}' if metric in ['revenue', 'spend'] else f'{x:,.0f}'),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{metric.replace("_", " ").title()} by Campaign',
                font=dict(size=18, color=self.COLORS['dark'])
            ),
            xaxis_title='Campaign',
            yaxis_title=metric.replace('_', ' ').title(),
            template='plotly_white',
            height=400,
            margin=dict(t=50, l=60, r=30, b=80),
            xaxis_tickangle=-45
        )
        
        filename = f'{self.output_dir}/campaign_{metric}.png'
        fig.write_image(filename, scale=2)
        
        return filename
    
    def create_platform_performance_chart(self, df: pl.DataFrame) -> str:
        """
        Create a pie chart showing performance by platform
        
        Args:
            df: DataFrame with platform data
            
        Returns:
            Path to saved chart image
        """
        if 'platform' not in df.columns:
            return None
        
        agg_df = df.group_by('platform').agg([
            pl.col('impressions').sum(),
            pl.col('revenue').sum()
        ])
        
        pdf = agg_df.to_pandas()
        
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]],
                           subplot_titles=('Impressions by Platform', 'Revenue by Platform'))
        
        fig.add_trace(go.Pie(
            labels=pdf['platform'],
            values=pdf['impressions'],
            marker_colors=self.COLOR_PALETTE,
            textinfo='percent+label',
            hole=0.4
        ), row=1, col=1)
        
        fig.add_trace(go.Pie(
            labels=pdf['platform'],
            values=pdf['revenue'],
            marker_colors=self.COLOR_PALETTE,
            textinfo='percent+label',
            hole=0.4
        ), row=1, col=2)
        
        fig.update_layout(
            title=dict(
                text='Performance by Platform',
                font=dict(size=18, color=self.COLORS['dark'])
            ),
            template='plotly_white',
            height=400,
            margin=dict(t=80, l=30, r=30, b=30),
            showlegend=False
        )
        
        filename = f'{self.output_dir}/platform_performance.png'
        fig.write_image(filename, scale=2)
        
        return filename
    
    def create_metrics_dashboard(self, df: pl.DataFrame) -> str:
        """
        Create a multi-metric dashboard view
        
        Args:
            df: DataFrame with metrics data
            
        Returns:
            Path to saved chart image
        """
        # Aggregate by date
        agg_df = df.group_by('date').agg([
            pl.col('impressions').sum(),
            pl.col('clicks').sum(),
            pl.col('conversions').sum(),
            pl.col('revenue').sum(),
            pl.col('spend').sum()
        ]).sort('date')
        
        pdf = agg_df.to_pandas()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Impressions & Clicks', 'Revenue vs Spend', 
                          'Conversions', 'ROI Trend'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Impressions & Clicks (dual axis)
        fig.add_trace(go.Scatter(
            x=pdf['date'], y=pdf['impressions'],
            name='Impressions', line=dict(color=self.COLORS['primary'], width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=pdf['date'], y=pdf['clicks'],
            name='Clicks', line=dict(color=self.COLORS['secondary'], width=2),
            yaxis='y2'
        ), row=1, col=1)
        
        # Revenue vs Spend
        fig.add_trace(go.Bar(
            x=pdf['date'], y=pdf['revenue'],
            name='Revenue', marker_color=self.COLORS['success']
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=pdf['date'], y=pdf['spend'],
            name='Spend', marker_color=self.COLORS['danger']
        ), row=1, col=2)
        
        # Conversions
        fig.add_trace(go.Scatter(
            x=pdf['date'], y=pdf['conversions'],
            name='Conversions', fill='tozeroy',
            line=dict(color=self.COLORS['info'], width=2),
            fillcolor='rgba(23, 162, 184, 0.3)'
        ), row=2, col=1)
        
        # ROI Trend
        pdf['roi'] = ((pdf['revenue'] - pdf['spend']) / pdf['spend'] * 100).round(2)
        fig.add_trace(go.Scatter(
            x=pdf['date'], y=pdf['roi'],
            name='ROI %', line=dict(color=self.COLORS['warning'], width=2),
            mode='lines+markers'
        ), row=2, col=2)
        
        fig.update_layout(
            title=dict(
                text='Performance Dashboard',
                font=dict(size=20, color=self.COLORS['dark'])
            ),
            template='plotly_white',
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=100, l=60, r=60, b=50)
        )
        
        filename = f'{self.output_dir}/dashboard.png'
        fig.write_image(filename, scale=2)
        
        return filename
    
    def create_anomaly_highlight_chart(self, df: pl.DataFrame, 
                                        anomalies: List[Dict],
                                        metric: str = 'impressions') -> str:
        """
        Create a chart highlighting anomalies
        
        Args:
            df: DataFrame with data
            anomalies: List of anomaly dictionaries
            metric: Metric to display
            
        Returns:
            Path to saved chart image
        """
        # Aggregate by date
        agg_df = df.group_by('date').agg(pl.col(metric).sum()).sort('date')
        pdf = agg_df.to_pandas()
        
        # Find anomaly dates for this metric
        anomaly_dates = [a['date'] for a in anomalies if a.get('metric') == metric]
        
        fig = go.Figure()
        
        # Main line
        fig.add_trace(go.Scatter(
            x=pdf['date'],
            y=pdf[metric],
            mode='lines+markers',
            line=dict(color=self.COLORS['primary'], width=2),
            marker=dict(size=6),
            name=metric.title()
        ))
        
        # Highlight anomalies
        if anomaly_dates:
            anomaly_df = pdf[pdf['date'].isin(anomaly_dates)]
            if len(anomaly_df) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_df['date'],
                    y=anomaly_df[metric],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=self.COLORS['danger'],
                        symbol='x',
                        line=dict(width=2, color='white')
                    ),
                    name='Anomaly'
                ))
        
        # Add average line
        avg_val = pdf[metric].mean()
        fig.add_hline(y=avg_val, line_dash="dash", line_color=self.COLORS['warning'],
                     annotation_text=f"Avg: {avg_val:,.0f}")
        
        fig.update_layout(
            title=dict(
                text=f'{metric.replace("_", " ").title()} with Anomalies Highlighted',
                font=dict(size=18, color=self.COLORS['dark'])
            ),
            xaxis_title='Date',
            yaxis_title=metric.replace('_', ' ').title(),
            template='plotly_white',
            height=400,
            margin=dict(t=50, l=60, r=30, b=50),
            showlegend=True
        )
        
        filename = f'{self.output_dir}/anomaly_{metric}.png'
        fig.write_image(filename, scale=2)
        
        return filename
    
    def create_week_over_week_chart(self, df: pl.DataFrame, 
                                     metric: str = 'revenue') -> str:
        """
        Create a week-over-week comparison chart
        
        Args:
            df: DataFrame with date and metric data
            metric: Metric to compare
            
        Returns:
            Path to saved chart image
        """
        # Aggregate by date
        agg_df = df.group_by('date').agg(pl.col(metric).sum()).sort('date')
        pdf = agg_df.to_pandas()
        
        # Calculate week-over-week change
        pdf['wow_change'] = pdf[metric].pct_change() * 100
        
        # Color based on positive/negative
        colors = [self.COLORS['success'] if x >= 0 else self.COLORS['danger'] 
                 for x in pdf['wow_change'].fillna(0)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=pdf['date'],
            y=pdf['wow_change'],
            marker_color=colors,
            text=pdf['wow_change'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else ''),
            textposition='outside'
        ))
        
        fig.add_hline(y=0, line_color=self.COLORS['dark'], line_width=1)
        
        fig.update_layout(
            title=dict(
                text=f'{metric.replace("_", " ").title()} - Day-over-Day Change',
                font=dict(size=18, color=self.COLORS['dark'])
            ),
            xaxis_title='Date',
            yaxis_title='Change (%)',
            template='plotly_white',
            height=400,
            margin=dict(t=50, l=60, r=30, b=80),
            xaxis_tickangle=-45
        )
        
        filename = f'{self.output_dir}/wow_{metric}.png'
        fig.write_image(filename, scale=2)
        
        return filename
    
    def generate_all_charts(self, df: pl.DataFrame, anomalies: List[Dict]) -> Dict[str, str]:
        """
        Generate all standard charts for the report
        
        Args:
            df: Main DataFrame
            anomalies: List of anomalies
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        charts = {}
        
        try:
            charts['dashboard'] = self.create_metrics_dashboard(df)
            print("  ✅ Dashboard chart created")
        except Exception as e:
            print(f"  ❌ Dashboard chart failed: {e}")
        
        try:
            charts['impressions_trend'] = self.create_time_series_chart(df, 'impressions')
            print("  ✅ Impressions trend chart created")
        except Exception as e:
            print(f"  ❌ Impressions trend failed: {e}")
        
        try:
            charts['revenue_by_campaign'] = self.create_campaign_comparison_chart(df, 'revenue')
            print("  ✅ Campaign comparison chart created")
        except Exception as e:
            print(f"  ❌ Campaign comparison failed: {e}")
        
        try:
            charts['platform_performance'] = self.create_platform_performance_chart(df)
            print("  ✅ Platform performance chart created")
        except Exception as e:
            print(f"  ❌ Platform performance failed: {e}")
        
        if anomalies:
            try:
                charts['anomaly_chart'] = self.create_anomaly_highlight_chart(df, anomalies, 'impressions')
                print("  ✅ Anomaly chart created")
            except Exception as e:
                print(f"  ❌ Anomaly chart failed: {e}")
        
        try:
            charts['wow_revenue'] = self.create_week_over_week_chart(df, 'revenue')
            print("  ✅ Week-over-week chart created")
        except Exception as e:
            print(f"  ❌ Week-over-week chart failed: {e}")
        
        return charts


# Need pandas for some Plotly operations
import pandas as pd

if __name__ == "__main__":
    # Test the module
    from data_ingestion import DataIngestion, DataTransformer
    
    ingestion = DataIngestion()
    transformer = DataTransformer()
    chart_gen = ChartGenerator()
    
    sample_path = "data/sample/ad_performance.csv"
    if os.path.exists(sample_path):
        df = ingestion.load_csv(sample_path)
        df = transformer.parse_dates(df, ['date'])
        df = transformer.calculate_metrics(df)
        
        print("\n📊 Generating Charts...")
        charts = chart_gen.generate_all_charts(df, [])
        
        print("\n✅ Charts generated:")
        for name, path in charts.items():
            print(f"   - {name}: {path}")
