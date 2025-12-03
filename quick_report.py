"""
TrendSpotter - Fast Pipeline (No Charts)
Generates PDF reports with AI insights in under 15 seconds
"""

import os
import time
from datetime import datetime
import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
from openai import OpenAI
from fpdf import FPDF

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "")
INPUT_FILE = "data/sample/ad_performance.csv"
OUTPUT_DIR = "output"

def main():
    start_time = time.time()
    
    print("=" * 50)
    print("TRENDSPOTTER - Fast Pipeline")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load Data
    print("\n[1/5] Loading Data...")
    df = pl.read_csv(INPUT_FILE)
    print(f"      Loaded {len(df)} records")
    
    # Step 2: Calculate Metrics
    print("[2/5] Calculating Metrics...")
    df = df.with_columns([
        (pl.col('clicks') / pl.col('impressions') * 100).round(2).alias('ctr'),
        ((pl.col('revenue') - pl.col('spend')) / pl.col('spend') * 100).round(2).alias('roi')
    ])
    
    # Step 3: Detect Anomalies
    print("[3/5] Detecting Anomalies...")
    features = df.select(['impressions', 'clicks', 'spend', 'revenue']).to_numpy()
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=50)
    predictions = iso_forest.fit_predict(features)
    df = df.with_columns(pl.Series('is_anomaly', predictions == -1))
    
    anomalies = []
    for row in df.filter(pl.col('is_anomaly')).iter_rows(named=True):
        anomalies.append({
            'campaign': row['campaign_name'],
            'date': str(row['date']),
            'region': row['region'],
            'desc': f"{row['campaign_name']} - unusual metrics on {row['date']} ({row['region']})"
        })
    print(f"      Found {len(anomalies)} anomalies")
    
    # Step 4: Calculate KPIs & Campaign Data
    print("[4/5] Calculating KPIs...")
    kpis = {
        'impressions': int(df['impressions'].sum()),
        'clicks': int(df['clicks'].sum()),
        'conversions': int(df['conversions'].sum()),
        'revenue': float(df['revenue'].sum()),
        'spend': float(df['spend'].sum()),
    }
    kpis['roi'] = ((kpis['revenue'] - kpis['spend']) / kpis['spend'] * 100)
    kpis['ctr'] = (kpis['clicks'] / kpis['impressions'] * 100)
    
    campaigns = df.group_by('campaign_name').agg([
        pl.col('impressions').sum(),
        pl.col('clicks').sum(),
        pl.col('revenue').sum(),
        pl.col('spend').sum()
    ]).sort('revenue', descending=True)
    
    dates = sorted(df['date'].unique().to_list())
    
    # Step 5: AI Analysis
    print("[5/5] AI Analysis...")
    client = OpenAI(api_key=API_KEY)
    
    context = f"""AdTech Data ({dates[0]} to {dates[-1]}):
- Impressions: {kpis['impressions']:,}
- Clicks: {kpis['clicks']:,}
- Conversions: {kpis['conversions']:,}
- Revenue: ${kpis['revenue']:,.0f}
- Spend: ${kpis['spend']:,.0f}
- ROI: {kpis['roi']:.1f}%
- Anomalies: {len(anomalies)}

Campaigns: {', '.join([f"{r['campaign_name']} (${r['revenue']:,.0f})" for r in campaigns.iter_rows(named=True)])}
Anomalies: {', '.join([a['desc'] for a in anomalies[:3]])}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Senior Data Analyst. Write 100-word executive summary with key insights."},
            {"role": "user", "content": context}
        ],
        max_tokens=200
    )
    summary = resp.choices[0].message.content
    
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Give exactly 3 one-sentence recommendations based on the data."},
            {"role": "user", "content": context}
        ],
        max_tokens=150
    )
    recs = [l.strip().lstrip('0123456789.-) ') for l in resp2.choices[0].message.content.split('\n') if l.strip()][:3]
    
    # Generate PDF
    print("\nGenerating PDF...")
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(46, 134, 171)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 8)
    pdf.cell(0, 10, 'Weekly Performance Report', 0, 1)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_xy(15, 20)
    pdf.cell(0, 8, f'{dates[0]} to {dates[-1]}', 0, 1)
    pdf.set_xy(15, 28)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%B %d, %Y %H:%M")}', 0, 1)
    
    # KPIs
    pdf.set_y(50)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'KEY METRICS', 0, 1)
    pdf.ln(2)
    
    kpi_list = [
        ('Total Impressions', f"{kpis['impressions']:,}"),
        ('Total Clicks', f"{kpis['clicks']:,}"),
        ('Total Conversions', f"{kpis['conversions']:,}"),
        ('Total Revenue', f"${kpis['revenue']:,.0f}"),
        ('Total Spend', f"${kpis['spend']:,.0f}"),
        ('Overall ROI', f"{kpis['roi']:.1f}%"),
    ]
    
    y = pdf.get_y()
    for i, (label, val) in enumerate(kpi_list):
        col, row = i % 3, i // 3
        x = 15 + col * 62
        yy = y + row * 22
        pdf.set_fill_color(245, 247, 250)
        pdf.rect(x, yy, 58, 18, 'F')
        pdf.set_fill_color(46, 134, 171)
        pdf.rect(x, yy, 2, 18, 'F')
        pdf.set_xy(x+5, yy+2)
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(50, 4, label.upper(), 0)
        pdf.set_xy(x+5, yy+8)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(50, 6, val, 0)
    
    # Executive Summary
    pdf.set_y(y + 52)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'EXECUTIVE SUMMARY (AI-Generated)', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 5, summary)
    
    # Anomalies
    if anomalies:
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(46, 134, 171)
        pdf.cell(0, 8, f'ANOMALIES DETECTED ({len(anomalies)})', 0, 1)
        for a in anomalies[:4]:
            yy = pdf.get_y()
            pdf.set_fill_color(255, 240, 240)
            pdf.rect(15, yy, 180, 8, 'F')
            pdf.set_fill_color(220, 53, 69)
            pdf.rect(15, yy, 2, 8, 'F')
            pdf.set_xy(20, yy+1.5)
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 5, a['desc'][:75], 0, 1)
            pdf.set_y(yy + 10)
    
    # Campaign Table
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'CAMPAIGN PERFORMANCE', 0, 1)
    
    pdf.set_fill_color(46, 134, 171)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 9)
    cols = [('Campaign', 50), ('Impressions', 32), ('Clicks', 28), ('Revenue', 32), ('ROI', 28)]
    x = 15
    for name, w in cols:
        pdf.set_xy(x, pdf.get_y())
        pdf.cell(w, 7, name, 1, 0, 'C', True)
        x += w
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(50, 50, 50)
    for i, row in enumerate(campaigns.iter_rows(named=True)):
        pdf.set_fill_color(250, 250, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        roi = ((row['revenue']-row['spend'])/row['spend']*100) if row['spend'] else 0
        data = [row['campaign_name'], f"{row['impressions']:,}", f"{row['clicks']:,}", f"${row['revenue']:,.0f}", f"{roi:.0f}%"]
        x = 15
        for (_, w), v in zip(cols, data):
            pdf.set_xy(x, pdf.get_y())
            pdf.cell(w, 6, v, 1, 0, 'C', True)
            x += w
        pdf.ln()
    
    # Recommendations
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'AI RECOMMENDATIONS', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    for r in recs:
        yy = pdf.get_y()
        pdf.set_fill_color(245, 250, 255)
        pdf.rect(15, yy, 180, 8, 'F')
        pdf.set_xy(18, yy+1.5)
        pdf.set_text_color(46, 134, 171)
        pdf.cell(5, 5, '>', 0)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(170, 5, r[:85], 0, 1)
        pdf.set_y(yy + 10)
    
    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f"{OUTPUT_DIR}/report_{ts}.pdf"
    pdf.output(path)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"DONE in {elapsed:.1f} seconds!")
    print(f"Output: {path}")
    print(f"{'='*50}")
    
    return path

if __name__ == "__main__":
    main()
