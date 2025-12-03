"""
TrendSpotter - Simplified Fast Pipeline
Generates PDF reports with AI insights in under 30 seconds
"""

import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
from openai import OpenAI
from fpdf import FPDF
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "")
INPUT_FILE = "data/sample/ad_performance.csv"
OUTPUT_DIR = "output"

def main():
    start_time = time.time()
    
    print("=" * 60)
    print("🚀 TRENDSPOTTER - Automated Insight Engine")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/charts", exist_ok=True)
    
    # Step 1: Load Data
    print("\n📥 Step 1: Loading Data...")
    df = pl.read_csv(INPUT_FILE)
    print(f"   ✅ Loaded {len(df)} records")
    
    # Step 2: Calculate Metrics
    print("\n🔄 Step 2: Calculating Metrics...")
    df = df.with_columns([
        (pl.col('clicks') / pl.col('impressions') * 100).round(2).alias('ctr'),
        (pl.col('conversions') / pl.col('clicks') * 100).round(2).alias('conversion_rate'),
        ((pl.col('revenue') - pl.col('spend')) / pl.col('spend') * 100).round(2).alias('roi')
    ])
    print("   ✅ Calculated CTR, Conversion Rate, ROI")
    
    # Step 3: Detect Anomalies
    print("\n🔍 Step 3: Detecting Anomalies...")
    features = df.select(['impressions', 'clicks', 'conversions', 'spend', 'revenue']).to_numpy()
    features = np.nan_to_num(features)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(features)
    
    df = df.with_columns(pl.Series('is_anomaly', predictions == -1))
    anomaly_count = int((predictions == -1).sum())
    print(f"   ✅ Found {anomaly_count} anomalies")
    
    # Get anomaly details
    anomalies = []
    anomaly_rows = df.filter(pl.col('is_anomaly'))
    for row in anomaly_rows.iter_rows(named=True):
        anomalies.append({
            'date': str(row['date']),
            'campaign_name': row['campaign_name'],
            'metric': 'impressions',
            'value': row['impressions'],
            'severity': 'high' if row['impressions'] < 50000 else 'medium',
            'description': f"{row['campaign_name']} had unusual activity on {row['date']} in {row['region']}"
        })
    
    # Step 4: Calculate KPIs
    print("\n📊 Step 4: Calculating KPIs...")
    kpis = {
        'total_impressions': int(df['impressions'].sum()),
        'total_clicks': int(df['clicks'].sum()),
        'total_conversions': int(df['conversions'].sum()),
        'total_revenue': float(df['revenue'].sum()),
        'total_spend': float(df['spend'].sum()),
    }
    kpis['roi'] = ((kpis['total_revenue'] - kpis['total_spend']) / kpis['total_spend'] * 100)
    kpis['ctr'] = (kpis['total_clicks'] / kpis['total_impressions'] * 100)
    
    print(f"   Total Revenue: ${kpis['total_revenue']:,.2f}")
    print(f"   Total Spend: ${kpis['total_spend']:,.2f}")
    print(f"   ROI: {kpis['roi']:.1f}%")
    
    # Step 5: Generate Charts
    print("\n📈 Step 5: Generating Charts...")
    
    # Daily metrics chart
    daily = df.group_by('date').agg([
        pl.col('impressions').sum(),
        pl.col('clicks').sum(),
        pl.col('revenue').sum(),
        pl.col('spend').sum()
    ]).sort('date')
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Daily Impressions', 'Daily Revenue vs Spend',
                                       'Daily Clicks', 'Daily ROI'))
    
    dates = daily['date'].to_list()
    
    fig.add_trace(go.Scatter(x=dates, y=daily['impressions'].to_list(), 
                            mode='lines+markers', name='Impressions',
                            line=dict(color='#2E86AB')), row=1, col=1)
    
    fig.add_trace(go.Bar(x=dates, y=daily['revenue'].to_list(), name='Revenue',
                        marker_color='#28A745'), row=1, col=2)
    fig.add_trace(go.Bar(x=dates, y=daily['spend'].to_list(), name='Spend',
                        marker_color='#DC3545'), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=dates, y=daily['clicks'].to_list(),
                            mode='lines+markers', name='Clicks',
                            line=dict(color='#A23B72')), row=2, col=1)
    
    daily_roi = [((r-s)/s*100) if s > 0 else 0 
                 for r, s in zip(daily['revenue'].to_list(), daily['spend'].to_list())]
    fig.add_trace(go.Scatter(x=dates, y=daily_roi, mode='lines+markers', 
                            name='ROI %', line=dict(color='#FFC107')), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True, title_text="Performance Dashboard")
    fig.write_image(f"{OUTPUT_DIR}/charts/dashboard.png", scale=2)
    print("   ✅ Dashboard chart created")
    
    # Campaign comparison
    campaigns = df.group_by('campaign_name').agg([
        pl.col('impressions').sum(),
        pl.col('clicks').sum(),
        pl.col('revenue').sum(),
        pl.col('spend').sum()
    ]).sort('revenue', descending=True)
    
    fig2 = go.Figure(data=[
        go.Bar(x=campaigns['campaign_name'].to_list(), 
               y=campaigns['revenue'].to_list(),
               marker_color=['#2E86AB', '#A23B72', '#28A745', '#FFC107', '#DC3545'])
    ])
    fig2.update_layout(title='Revenue by Campaign', height=400)
    fig2.write_image(f"{OUTPUT_DIR}/charts/campaign_revenue.png", scale=2)
    print("   ✅ Campaign chart created")
    
    # Step 6: AI Analysis
    print("\n🤖 Step 6: Generating AI Analysis...")
    
    client = OpenAI(api_key=API_KEY)
    
    context = f"""
AdTech Performance Data Summary:
- Period: {dates[0]} to {dates[-1]}
- Total Impressions: {kpis['total_impressions']:,}
- Total Clicks: {kpis['total_clicks']:,}
- Total Conversions: {kpis['total_conversions']:,}
- Total Revenue: ${kpis['total_revenue']:,.2f}
- Total Spend: ${kpis['total_spend']:,.2f}
- Overall ROI: {kpis['roi']:.1f}%
- CTR: {kpis['ctr']:.2f}%
- Anomalies Detected: {len(anomalies)}

Top Campaigns by Revenue:
{chr(10).join([f"- {row['campaign_name']}: ${row['revenue']:,.0f}" for row in campaigns.iter_rows(named=True)])}

Anomalies:
{chr(10).join([f"- {a['description']}" for a in anomalies[:5]])}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Senior Data Analyst. Write a brief executive summary (150 words max) of the AdTech performance. Be specific with numbers. Highlight key insights and concerns."},
            {"role": "user", "content": context}
        ],
        max_tokens=300,
        temperature=0.3
    )
    
    executive_summary = response.choices[0].message.content
    print("   ✅ Executive summary generated")
    
    # Get recommendations
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AdTech consultant. Based on the data, provide exactly 4 specific actionable recommendations. Each should be one sentence."},
            {"role": "user", "content": context}
        ],
        max_tokens=200,
        temperature=0.3
    )
    
    recommendations = [line.strip().lstrip('0123456789.-) ') 
                       for line in response2.choices[0].message.content.split('\n') 
                       if line.strip() and len(line.strip()) > 10][:4]
    print("   ✅ Recommendations generated")
    
    # Step 7: Generate PDF
    print("\n📄 Step 7: Generating PDF Report...")
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1 - Header and KPIs
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(46, 134, 171)
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 10)
    pdf.cell(0, 10, 'Weekly Performance Report', 0, 1)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_xy(15, 22)
    pdf.cell(0, 10, f'Period: {dates[0]} to {dates[-1]}', 0, 1)
    pdf.set_xy(15, 32)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%B %d, %Y %H:%M")}', 0, 1)
    
    # KPIs
    pdf.set_y(55)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 10, 'Key Performance Indicators', 0, 1)
    
    kpi_items = [
        ('Impressions', f"{kpis['total_impressions']:,}"),
        ('Clicks', f"{kpis['total_clicks']:,}"),
        ('Conversions', f"{kpis['total_conversions']:,}"),
        ('Revenue', f"${kpis['total_revenue']:,.0f}"),
        ('Spend', f"${kpis['total_spend']:,.0f}"),
        ('ROI', f"{kpis['roi']:.1f}%"),
    ]
    
    start_y = pdf.get_y() + 5
    for i, (label, value) in enumerate(kpi_items):
        col = i % 3
        row = i // 3
        x = 15 + col * 62
        y = start_y + row * 25
        
        pdf.set_fill_color(248, 249, 250)
        pdf.rect(x, y, 58, 22, 'F')
        pdf.set_fill_color(46, 134, 171)
        pdf.rect(x, y, 2, 22, 'F')
        
        pdf.set_xy(x + 5, y + 3)
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(50, 5, label.upper(), 0, 0)
        
        pdf.set_xy(x + 5, y + 10)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(52, 58, 64)
        pdf.cell(50, 8, value, 0, 0)
    
    # Executive Summary
    pdf.set_y(start_y + 60)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 10, 'Executive Summary (AI-Generated)', 0, 1)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(52, 58, 64)
    pdf.multi_cell(0, 5, executive_summary)
    
    # Anomalies
    if anomalies:
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(46, 134, 171)
        pdf.cell(0, 10, f'Anomaly Alerts ({len(anomalies)} detected)', 0, 1)
        
        for a in anomalies[:3]:
            y = pdf.get_y()
            color = (220, 53, 69) if a['severity'] == 'high' else (255, 193, 7)
            pdf.set_fill_color(255, 245, 245)
            pdf.rect(15, y, 180, 12, 'F')
            pdf.set_fill_color(*color)
            pdf.rect(15, y, 3, 12, 'F')
            
            pdf.set_xy(20, y + 3)
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(52, 58, 64)
            pdf.cell(0, 5, a['description'][:70], 0, 1)
            pdf.set_y(y + 15)
    
    # Page 2 - Charts
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 10, 'Performance Visualizations', 0, 1)
    
    if os.path.exists(f"{OUTPUT_DIR}/charts/dashboard.png"):
        pdf.image(f"{OUTPUT_DIR}/charts/dashboard.png", x=10, y=25, w=190)
    
    if os.path.exists(f"{OUTPUT_DIR}/charts/campaign_revenue.png"):
        pdf.image(f"{OUTPUT_DIR}/charts/campaign_revenue.png", x=10, y=145, w=190)
    
    # Page 3 - Campaign Table & Recommendations
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 10, 'Campaign Performance', 0, 1)
    
    # Table header
    pdf.set_fill_color(46, 134, 171)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 9)
    cols = [('Campaign', 45), ('Impressions', 30), ('Clicks', 25), ('Revenue', 30), ('Spend', 30), ('ROI', 25)]
    x = 15
    for name, width in cols:
        pdf.set_xy(x, pdf.get_y())
        pdf.cell(width, 8, name, 1, 0, 'C', True)
        x += width
    pdf.ln()
    
    # Table rows
    pdf.set_text_color(52, 58, 64)
    pdf.set_font('Helvetica', '', 9)
    for i, row in enumerate(campaigns.iter_rows(named=True)):
        if i % 2 == 0:
            pdf.set_fill_color(248, 249, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        
        roi = ((row['revenue'] - row['spend']) / row['spend'] * 100) if row['spend'] > 0 else 0
        data = [
            row['campaign_name'][:15],
            f"{row['impressions']:,}",
            f"{row['clicks']:,}",
            f"${row['revenue']:,.0f}",
            f"${row['spend']:,.0f}",
            f"{roi:.1f}%"
        ]
        x = 15
        for (_, width), val in zip(cols, data):
            pdf.set_xy(x, pdf.get_y())
            pdf.cell(width, 7, val, 1, 0, 'C', True)
            x += width
        pdf.ln()
    
    # Recommendations
    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 10, 'AI Recommendations', 0, 1)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(52, 58, 64)
    for i, rec in enumerate(recommendations, 1):
        y = pdf.get_y()
        pdf.set_fill_color(248, 249, 250)
        pdf.rect(15, y, 180, 10, 'F')
        pdf.set_xy(18, y + 2)
        pdf.set_text_color(46, 134, 171)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(5, 6, '>', 0, 0)
        pdf.set_text_color(52, 58, 64)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(170, 6, rec[:80], 0, 1)
        pdf.set_y(y + 13)
    
    # Save PDF
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = f"{OUTPUT_DIR}/report_{timestamp}.pdf"
    pdf.output(pdf_path)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"✅ PIPELINE COMPLETE!")
    print(f"{'=' * 60}")
    print(f"   Duration: {elapsed:.2f} seconds")
    print(f"   Records: {len(df):,}")
    print(f"   Anomalies: {len(anomalies)}")
    print(f"   Output: {pdf_path}")
    print(f"{'=' * 60}")
    
    return pdf_path

if __name__ == "__main__":
    main()
