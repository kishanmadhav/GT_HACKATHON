"""
TrendSpotter - Main Pipeline Orchestration
Automated Insight Engine for AdTech Data

This module ties together all components:
- Data Ingestion (CSV/JSON/Parquet)
- Anomaly Detection (Isolation Forest)
- AI Analysis (GPT-4o)
- Visualization (Plotly)
- Report Generation (PDF via WeasyPrint)
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import DataIngestion, DataTransformer
from anomaly_detection import AnomalyDetector
from ai_analyzer import AIAnalyzer
from visualization import ChartGenerator
from report_generator import ReportGenerator, KPIData
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TrendSpotterPipeline:
    """
    Main pipeline orchestrator for TrendSpotter
    
    Processes raw data files and generates executive-ready PDF reports
    with AI-generated insights in under 30 seconds
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the pipeline components
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.charts_dir = os.path.join(output_dir, 'charts')
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Initialize components
        print("🚀 Initializing TrendSpotter Pipeline...")
        
        self.ingestion = DataIngestion()
        self.transformer = DataTransformer()
        self.detector = AnomalyDetector(contamination=0.1)
        
        # Initialize AI analyzer with API key
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.ai_analyzer = AIAnalyzer(api_key=api_key)
            print("   ✅ AI Analyzer initialized (GPT-4o)")
        else:
            self.ai_analyzer = None
            print("   ⚠️ AI Analyzer not available (no API key)")
        
        self.chart_generator = ChartGenerator(output_dir=self.charts_dir)
        self.report_generator = ReportGenerator(
            template_dir='templates',
            output_dir=output_dir
        )
        
        print("   ✅ All components initialized\n")
    
    def process_file(self, file_path: str) -> str:
        """
        Process a single data file through the entire pipeline
        
        Args:
            file_path: Path to the input data file
            
        Returns:
            Path to the generated PDF report
        """
        start_time = time.time()
        
        print(f"{'='*60}")
        print(f"📊 TRENDSPOTTER PIPELINE")
        print(f"{'='*60}")
        print(f"Input: {file_path}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Step 1: Data Ingestion
        print("📥 Step 1: Data Ingestion")
        print("-" * 40)
        df = self.ingestion.load_auto(file_path)
        
        # Step 2: Data Transformation
        print("\n🔄 Step 2: Data Transformation")
        print("-" * 40)
        df = self.transformer.parse_dates(df, ['date'])
        df = self.transformer.calculate_metrics(df)
        print(f"   ✅ Calculated metrics: CTR, Conversion Rate, CPA, ROI, ROAS")
        
        # Step 3: Anomaly Detection
        print("\n🔍 Step 3: Anomaly Detection")
        print("-" * 40)
        df, anomalies = self.detector.detect(df)
        anomaly_list = self.detector.to_dict()
        
        # Step 4: Generate Visualizations
        print("\n📈 Step 4: Generating Charts")
        print("-" * 40)
        charts = self.chart_generator.generate_all_charts(df, anomaly_list)
        
        # Step 5: Calculate KPIs
        print("\n📊 Step 5: Calculating KPIs")
        print("-" * 40)
        kpis = self._calculate_kpis(df)
        campaign_data = self._get_campaign_summary(df)
        
        # Step 6: AI Analysis
        print("\n🤖 Step 6: AI Analysis")
        print("-" * 40)
        if self.ai_analyzer:
            data_summary = self._prepare_data_summary(df, kpis)
            campaign_perf = {c['name']: {'revenue': c['revenue'], 'roi': c['roi']} 
                           for c in campaign_data}
            
            print("   Generating executive summary...")
            executive_summary = self.ai_analyzer.generate_executive_summary(
                data_summary, anomaly_list, campaign_perf
            )
            
            print("   Generating anomaly analysis...")
            anomaly_analysis = self.ai_analyzer.generate_anomaly_analysis(anomaly_list)
            
            print("   Generating recommendations...")
            recommendations = self.ai_analyzer.generate_recommendations(
                data_summary, anomaly_list
            )
        else:
            executive_summary = self._generate_fallback_summary(kpis)
            anomaly_analysis = ""
            recommendations = self._get_default_recommendations()
        
        # Step 7: Generate Report
        print("\n📄 Step 7: Generating PDF Report")
        print("-" * 40)
        
        # Prepare report data
        report_data = {
            'title': 'Weekly Performance Report',
            'subtitle': self._get_date_range(df),
            'kpis': kpis,
            'executive_summary': executive_summary,
            'anomalies': anomaly_list,
            'anomaly_analysis': anomaly_analysis,
            'charts': {k: os.path.abspath(v) for k, v in charts.items() if v},
            'campaign_data': campaign_data,
            'recommendations': recommendations
        }
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_name = Path(file_path).stem
        output_filename = f'report_{input_name}_{timestamp}'
        
        pdf_path = self.report_generator.generate_report(report_data, output_filename)
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"   Duration: {elapsed:.2f} seconds")
        print(f"   Records processed: {len(df):,}")
        print(f"   Anomalies detected: {len(anomaly_list)}")
        print(f"   Output: {pdf_path}")
        print(f"{'='*60}\n")
        
        return pdf_path
    
    def _calculate_kpis(self, df) -> KPIData:
        """Calculate KPI metrics from the data"""
        total_impressions = int(df['impressions'].sum()) if 'impressions' in df.columns else 0
        total_clicks = int(df['clicks'].sum()) if 'clicks' in df.columns else 0
        total_conversions = int(df['conversions'].sum()) if 'conversions' in df.columns else 0
        total_revenue = float(df['revenue'].sum()) if 'revenue' in df.columns else 0
        total_spend = float(df['spend'].sum()) if 'spend' in df.columns else 0
        
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        roi = ((total_revenue - total_spend) / total_spend * 100) if total_spend > 0 else 0
        
        print(f"   Total Impressions: {total_impressions:,}")
        print(f"   Total Clicks: {total_clicks:,}")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   ROI: {roi:.1f}%")
        
        return KPIData(
            total_impressions=total_impressions,
            total_clicks=total_clicks,
            total_conversions=total_conversions,
            total_revenue=total_revenue,
            total_spend=total_spend,
            roi=roi,
            ctr=ctr,
            conversion_rate=conversion_rate
        )
    
    def _get_campaign_summary(self, df) -> list:
        """Get summary data for each campaign"""
        import polars as pl
        
        if 'campaign_name' not in df.columns:
            return []
        
        campaigns = df.group_by('campaign_name').agg([
            pl.col('impressions').sum().alias('impressions'),
            pl.col('clicks').sum().alias('clicks'),
            pl.col('conversions').sum().alias('conversions'),
            pl.col('spend').sum().alias('spend'),
            pl.col('revenue').sum().alias('revenue')
        ])
        
        result = []
        for row in campaigns.iter_rows(named=True):
            impressions = row['impressions'] or 0
            clicks = row['clicks'] or 0
            revenue = row['revenue'] or 0
            spend = row['spend'] or 0
            
            ctr = (clicks / impressions * 100) if impressions > 0 else 0
            roi = ((revenue - spend) / spend * 100) if spend > 0 else 0
            
            result.append({
                'name': row['campaign_name'],
                'impressions': impressions,
                'clicks': clicks,
                'ctr': round(ctr, 2),
                'revenue': revenue,
                'roi': round(roi, 1)
            })
        
        # Sort by revenue descending
        result.sort(key=lambda x: x['revenue'], reverse=True)
        return result
    
    def _prepare_data_summary(self, df, kpis) -> dict:
        """Prepare data summary for AI analysis"""
        return {
            'total_records': len(df),
            'date_range': self._get_date_range(df),
            'total_impressions': kpis.total_impressions,
            'total_clicks': kpis.total_clicks,
            'total_conversions': kpis.total_conversions,
            'total_spend': kpis.total_spend,
            'total_revenue': kpis.total_revenue,
            'avg_ctr': kpis.ctr,
            'avg_roi': kpis.roi
        }
    
    def _get_date_range(self, df) -> str:
        """Get the date range from the data"""
        if 'date' not in df.columns:
            return 'Date range not available'
        
        try:
            min_date = df['date'].min()
            max_date = df['date'].max()
            return f"{min_date} to {max_date}"
        except:
            return 'Date range not available'
    
    def _generate_fallback_summary(self, kpis) -> str:
        """Generate a basic summary when AI is not available"""
        return f"""## Performance Overview

This automated report provides an analysis of your advertising campaign performance.

**Key Metrics:**
- Total Impressions: {kpis.total_impressions:,}
- Total Clicks: {kpis.total_clicks:,}
- Total Conversions: {kpis.total_conversions:,}
- Total Revenue: ${kpis.total_revenue:,.2f}
- Total Spend: ${kpis.total_spend:,.2f}
- Overall ROI: {kpis.roi:.1f}%

**Summary:**
The campaigns generated ${kpis.total_revenue:,.2f} in revenue against ${kpis.total_spend:,.2f} in ad spend, 
resulting in a {kpis.roi:.1f}% return on investment.

Please review the detailed charts and anomaly analysis below for more insights.
"""
    
    def _get_default_recommendations(self) -> list:
        """Return default recommendations when AI is not available"""
        return [
            "Review campaigns with detected anomalies for potential optimization",
            "Monitor underperforming campaigns closely over the next week",
            "Consider A/B testing ad creatives for campaigns with declining CTR",
            "Analyze high-performing campaigns to replicate success factors",
            "Review budget allocation based on ROI performance"
        ]


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TrendSpotter - Automated Insight Engine for AdTech',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py data/sample/ad_performance.csv
  python main.py --watch data/input
  python main.py --output reports data/sample/ad_performance.csv
        '''
    )
    
    parser.add_argument('input', nargs='?', help='Input data file (CSV/JSON/Parquet)')
    parser.add_argument('--watch', '-w', metavar='DIR', 
                        help='Watch a directory for new files')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory for reports (default: output)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrendSpotterPipeline(output_dir=args.output)
    
    if args.watch:
        # Watch mode
        from file_watcher import FolderWatcher
        
        watcher = FolderWatcher(args.watch, pipeline.process_file)
        watcher.process_existing_files()
        watcher.start(blocking=True)
    
    elif args.input:
        # Process single file
        if not os.path.exists(args.input):
            print(f"❌ Error: File not found: {args.input}")
            sys.exit(1)
        
        pipeline.process_file(args.input)
    
    else:
        # No arguments - use sample data
        sample_file = 'data/sample/ad_performance.csv'
        
        if os.path.exists(sample_file):
            print("ℹ️  No input specified, using sample data...")
            pipeline.process_file(sample_file)
        else:
            print("❌ Error: No input file specified and no sample data found")
            print("   Usage: python main.py <input_file>")
            print("   Or:    python main.py --watch <directory>")
            sys.exit(1)


if __name__ == "__main__":
    main()
