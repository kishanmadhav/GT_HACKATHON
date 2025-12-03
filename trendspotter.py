"""
TrendSpotter - Complete Pipeline
Generates PDF or PowerPoint reports with AI insights and charts
Usage: python trendspotter.py [--pdf | --pptx]
"""

import os
import sys
import time
from datetime import datetime
import polars as pl
import numpy as np
from sklearn.ensemble import IsolationForest
from openai import OpenAI
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "")
INPUT_FILE = "data/sample/ad_performance.csv"
OUTPUT_DIR = "output"
CHARTS_DIR = "output/charts"

# Colors
BLUE = '#2E86AB'
PURPLE = '#A23B72'
GREEN = '#28A745'
RED = '#DC3545'
ORANGE = '#FFC107'


def generate_charts(df, dates, campaigns):
    """Generate charts using matplotlib (fast)"""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    
    # Daily aggregation
    daily = df.group_by('date').agg([
        pl.col('impressions').sum(),
        pl.col('clicks').sum(),
        pl.col('revenue').sum(),
        pl.col('spend').sum()
    ]).sort('date')
    
    d_dates = daily['date'].to_list()
    d_imp = daily['impressions'].to_list()
    d_rev = daily['revenue'].to_list()
    d_spend = daily['spend'].to_list()
    d_clicks = daily['clicks'].to_list()
    
    # Chart 1: Dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Performance Dashboard', fontsize=16, fontweight='bold', color=BLUE)
    
    # Impressions trend
    axes[0, 0].fill_between(range(len(d_dates)), d_imp, alpha=0.3, color=BLUE)
    axes[0, 0].plot(d_imp, color=BLUE, linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Daily Impressions', fontweight='bold')
    axes[0, 0].set_xticks(range(0, len(d_dates), 3))
    axes[0, 0].set_xticklabels([d_dates[i][-5:] for i in range(0, len(d_dates), 3)], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Revenue vs Spend
    x = range(len(d_dates))
    width = 0.35
    axes[0, 1].bar([i - width/2 for i in x], d_rev, width, label='Revenue', color=GREEN)
    axes[0, 1].bar([i + width/2 for i in x], d_spend, width, label='Spend', color=RED)
    axes[0, 1].set_title('Revenue vs Spend', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_xticks(range(0, len(d_dates), 3))
    axes[0, 1].set_xticklabels([d_dates[i][-5:] for i in range(0, len(d_dates), 3)], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Clicks trend
    axes[1, 0].fill_between(range(len(d_dates)), d_clicks, alpha=0.3, color=PURPLE)
    axes[1, 0].plot(d_clicks, color=PURPLE, linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_title('Daily Clicks', fontweight='bold')
    axes[1, 0].set_xticks(range(0, len(d_dates), 3))
    axes[1, 0].set_xticklabels([d_dates[i][-5:] for i in range(0, len(d_dates), 3)], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROI trend
    d_roi = [((r-s)/s*100) if s > 0 else 0 for r, s in zip(d_rev, d_spend)]
    axes[1, 1].plot(d_roi, color=ORANGE, linewidth=2, marker='s', markersize=4)
    axes[1, 1].axhline(y=np.mean(d_roi), color='gray', linestyle='--', label=f'Avg: {np.mean(d_roi):.0f}%')
    axes[1, 1].set_title('Daily ROI %', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_xticks(range(0, len(d_dates), 3))
    axes[1, 1].set_xticklabels([d_dates[i][-5:] for i in range(0, len(d_dates), 3)], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Campaign Revenue Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    camp_names = campaigns['campaign_name'].to_list()
    camp_rev = campaigns['revenue'].to_list()
    colors = [BLUE, PURPLE, GREEN, ORANGE, RED][:len(camp_names)]
    bars = ax.bar(camp_names, camp_rev, color=colors)
    ax.set_title('Revenue by Campaign', fontsize=14, fontweight='bold', color=BLUE)
    ax.set_ylabel('Revenue ($)')
    for bar, val in zip(bars, camp_rev):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/campaign_revenue.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Platform Pie Chart
    if 'platform' in df.columns:
        platforms = df.group_by('platform').agg(pl.col('revenue').sum()).sort('revenue', descending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(platforms['revenue'].to_list(), labels=platforms['platform'].to_list(),
               autopct='%1.1f%%', colors=[BLUE, PURPLE, GREEN, ORANGE, RED], startangle=90)
        ax.set_title('Revenue by Platform', fontsize=14, fontweight='bold', color=BLUE)
        plt.tight_layout()
        plt.savefig(f'{CHARTS_DIR}/platform_pie.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'dashboard': f'{CHARTS_DIR}/dashboard.png',
        'campaign': f'{CHARTS_DIR}/campaign_revenue.png',
        'platform': f'{CHARTS_DIR}/platform_pie.png'
    }


def generate_pdf(data):
    """Generate PDF report"""
    pdf = FPDF()
    
    # Page 1: Header, KPIs, Summary
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(46, 134, 171)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 8)
    pdf.cell(0, 10, 'Weekly Performance Report')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_xy(15, 20)
    pdf.cell(0, 8, f"{data['dates'][0]} to {data['dates'][-1]}")
    pdf.set_xy(15, 28)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}")
    
    # KPIs
    pdf.set_y(50)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'KEY METRICS')
    pdf.ln(10)
    
    kpi_list = [
        ('Impressions', f"{data['kpis']['impressions']:,}"),
        ('Clicks', f"{data['kpis']['clicks']:,}"),
        ('Conversions', f"{data['kpis']['conversions']:,}"),
        ('Revenue', f"${data['kpis']['revenue']:,.0f}"),
        ('Spend', f"${data['kpis']['spend']:,.0f}"),
        ('ROI', f"{data['kpis']['roi']:.1f}%"),
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
        pdf.cell(50, 4, label.upper())
        pdf.set_xy(x+5, yy+8)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(50, 6, val)
    
    # Executive Summary
    pdf.set_y(y + 52)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'EXECUTIVE SUMMARY (AI-Generated)')
    pdf.ln(8)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 5, data['summary'])
    
    # Anomalies
    if data['anomalies']:
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(46, 134, 171)
        pdf.cell(0, 8, f"ANOMALIES ({len(data['anomalies'])})")
        pdf.ln(8)
        for a in data['anomalies'][:4]:
            yy = pdf.get_y()
            pdf.set_fill_color(255, 240, 240)
            pdf.rect(15, yy, 180, 8, 'F')
            pdf.set_fill_color(220, 53, 69)
            pdf.rect(15, yy, 2, 8, 'F')
            pdf.set_xy(20, yy+1.5)
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 5, a['desc'][:75])
            pdf.set_y(yy + 10)
    
    # Page 2: Charts
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'PERFORMANCE CHARTS')
    pdf.ln(10)
    
    if os.path.exists(data['charts']['dashboard']):
        pdf.image(data['charts']['dashboard'], x=10, y=30, w=190)
    
    # Page 3: Campaign Chart + Table + Recommendations
    pdf.add_page()
    
    if os.path.exists(data['charts']['campaign']):
        pdf.image(data['charts']['campaign'], x=15, y=15, w=180)
    
    pdf.set_y(95)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'CAMPAIGN PERFORMANCE')
    pdf.ln(8)
    
    # Table
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
    for i, row in enumerate(data['campaigns'].iter_rows(named=True)):
        pdf.set_fill_color(250, 250, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        roi = ((row['revenue']-row['spend'])/row['spend']*100) if row['spend'] else 0
        vals = [row['campaign_name'], f"{row['impressions']:,}", f"{row['clicks']:,}", f"${row['revenue']:,.0f}", f"{roi:.0f}%"]
        x = 15
        for (_, w), v in zip(cols, vals):
            pdf.set_xy(x, pdf.get_y())
            pdf.cell(w, 6, v, 1, 0, 'C', True)
            x += w
        pdf.ln()
    
    # Recommendations
    pdf.ln(8)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, 'AI RECOMMENDATIONS')
    pdf.ln(8)
    pdf.set_font('Helvetica', '', 10)
    for r in data['recommendations']:
        yy = pdf.get_y()
        pdf.set_fill_color(245, 250, 255)
        pdf.rect(15, yy, 180, 8, 'F')
        pdf.set_xy(18, yy+1.5)
        pdf.set_text_color(46, 134, 171)
        pdf.cell(5, 5, '>')
        pdf.set_text_color(50, 50, 50)
        pdf.cell(170, 5, r[:85])
        pdf.set_y(yy + 10)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f"{OUTPUT_DIR}/report_{ts}.pdf"
    pdf.output(path)
    return path


def generate_pptx(data):
    """Generate PowerPoint report"""
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    
    # Slide 1: Title
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    
    # Title background
    shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(2.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(46, 134, 171)
    shape.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(12), Inches(1))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Weekly Performance Report"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    subtitle = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(12), Inches(0.5))
    tf = subtitle.text_frame
    p = tf.paragraphs[0]
    p.text = f"{data['dates'][0]} to {data['dates'][-1]} | Generated: {datetime.now().strftime('%B %d, %Y')}"
    p.font.size = Pt(20)
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    # KPIs on title slide
    kpis = data['kpis']
    kpi_data = [
        ('Impressions', f"{kpis['impressions']:,}"),
        ('Clicks', f"{kpis['clicks']:,}"),
        ('Conversions', f"{kpis['conversions']:,}"),
        ('Revenue', f"${kpis['revenue']:,.0f}"),
        ('Spend', f"${kpis['spend']:,.0f}"),
        ('ROI', f"{kpis['roi']:.1f}%"),
    ]
    
    for i, (label, val) in enumerate(kpi_data):
        col = i % 3
        row = i // 3
        x = Inches(0.8 + col * 4.2)
        y = Inches(3 + row * 1.8)
        
        box = slide.shapes.add_shape(1, x, y, Inches(3.8), Inches(1.5))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(245, 247, 250)
        box.line.fill.background()
        
        lbl = slide.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(3.4), Inches(0.4))
        tf = lbl.text_frame
        p = tf.paragraphs[0]
        p.text = label.upper()
        p.font.size = Pt(12)
        p.font.color.rgb = RGBColor(100, 100, 100)
        
        v = slide.shapes.add_textbox(x + Inches(0.2), y + Inches(0.6), Inches(3.4), Inches(0.7))
        tf = v.text_frame
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = RGBColor(46, 134, 171)
    
    # Slide 2: Executive Summary
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Executive Summary"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RGBColor(46, 134, 171)
    
    badge = slide.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(2), Inches(0.4))
    tf = badge.text_frame
    p = tf.paragraphs[0]
    p.text = "🤖 AI-Generated"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(100, 100, 100)
    
    summary = slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(12), Inches(3))
    tf = summary.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = data['summary']
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(50, 50, 50)
    
    # Anomalies
    if data['anomalies']:
        anom_title = slide.shapes.add_textbox(Inches(0.5), Inches(4.8), Inches(12), Inches(0.5))
        tf = anom_title.text_frame
        p = tf.paragraphs[0]
        p.text = f"⚠️ Anomalies Detected: {len(data['anomalies'])}"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = RGBColor(220, 53, 69)
        
        for i, a in enumerate(data['anomalies'][:3]):
            anom = slide.shapes.add_textbox(Inches(0.7), Inches(5.4 + i*0.5), Inches(11), Inches(0.4))
            tf = anom.text_frame
            p = tf.paragraphs[0]
            p.text = f"• {a['desc'][:80]}"
            p.font.size = Pt(14)
            p.font.color.rgb = RGBColor(80, 80, 80)
    
    # Slide 3: Dashboard Chart
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Performance Dashboard"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RGBColor(46, 134, 171)
    
    if os.path.exists(data['charts']['dashboard']):
        slide.shapes.add_picture(data['charts']['dashboard'], Inches(0.5), Inches(1.2), width=Inches(12))
    
    # Slide 4: Campaign Performance
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Campaign Performance"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RGBColor(46, 134, 171)
    
    if os.path.exists(data['charts']['campaign']):
        slide.shapes.add_picture(data['charts']['campaign'], Inches(0.5), Inches(1.2), width=Inches(12))
    
    # Slide 5: Recommendations
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "AI Recommendations"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RGBColor(46, 134, 171)
    
    for i, rec in enumerate(data['recommendations']):
        box = slide.shapes.add_shape(1, Inches(0.5), Inches(1.5 + i*1.3), Inches(12), Inches(1))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(245, 250, 255)
        box.line.fill.background()
        
        txt = slide.shapes.add_textbox(Inches(0.8), Inches(1.7 + i*1.3), Inches(11.4), Inches(0.8))
        tf = txt.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"→ {rec}"
        p.font.size = Pt(20)
        p.font.color.rgb = RGBColor(50, 50, 50)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f"{OUTPUT_DIR}/report_{ts}.pptx"
    prs.save(path)
    return path


def main():
    start_time = time.time()
    
    # Check command line args
    output_format = 'pdf'  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--pptx', '-p', 'pptx', 'ppt']:
            output_format = 'pptx'
        elif sys.argv[1] in ['--pdf', 'pdf']:
            output_format = 'pdf'
        elif sys.argv[1] in ['--help', '-h']:
            print("TrendSpotter - Automated Report Generator")
            print("\nUsage: python trendspotter.py [--pdf | --pptx]")
            print("\nOptions:")
            print("  --pdf   Generate PDF report (default)")
            print("  --pptx  Generate PowerPoint report")
            return
    
    print("=" * 50)
    print(f"TRENDSPOTTER - Generating {output_format.upper()}")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    
    # Step 1: Load Data
    print("\n[1/6] Loading Data...")
    df = pl.read_csv(INPUT_FILE)
    print(f"      ✓ {len(df)} records loaded")
    
    # Step 2: Calculate Metrics
    print("[2/6] Calculating Metrics...")
    df = df.with_columns([
        (pl.col('clicks') / pl.col('impressions') * 100).round(2).alias('ctr'),
        ((pl.col('revenue') - pl.col('spend')) / pl.col('spend') * 100).round(2).alias('roi')
    ])
    print("      ✓ CTR, ROI calculated")
    
    # Step 3: Detect Anomalies
    print("[3/6] Detecting Anomalies...")
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
    print(f"      ✓ {len(anomalies)} anomalies found")
    
    # Step 4: Calculate KPIs & Aggregations
    print("[4/6] Aggregating Data...")
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
    print(f"      ✓ KPIs calculated, {len(campaigns)} campaigns")
    
    # Step 5: Generate Charts
    print("[5/6] Generating Charts...")
    charts = generate_charts(df, dates, campaigns)
    print("      ✓ Charts generated")
    
    # Step 6: AI Analysis
    print("[6/6] AI Analysis...")
    client = OpenAI(api_key=API_KEY)
    
    context = f"""AdTech Data ({dates[0]} to {dates[-1]}):
- Impressions: {kpis['impressions']:,}, Clicks: {kpis['clicks']:,}, Conversions: {kpis['conversions']:,}
- Revenue: ${kpis['revenue']:,.0f}, Spend: ${kpis['spend']:,.0f}, ROI: {kpis['roi']:.1f}%
- Anomalies: {len(anomalies)}
Campaigns: {', '.join([f"{r['campaign_name']} (${r['revenue']:,.0f})" for r in campaigns.iter_rows(named=True)])}"""

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
            {"role": "system", "content": "Give exactly 4 one-sentence recommendations."},
            {"role": "user", "content": context}
        ],
        max_tokens=200
    )
    recs = [l.strip().lstrip('0123456789.-) ') for l in resp2.choices[0].message.content.split('\n') if l.strip() and len(l) > 10][:4]
    print("      ✓ AI analysis complete")
    
    # Prepare data for report
    report_data = {
        'kpis': kpis,
        'dates': dates,
        'campaigns': campaigns,
        'anomalies': anomalies,
        'summary': summary,
        'recommendations': recs,
        'charts': charts
    }
    
    # Generate Report
    print(f"\nGenerating {output_format.upper()} report...")
    if output_format == 'pptx':
        path = generate_pptx(report_data)
    else:
        path = generate_pdf(report_data)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"✅ DONE in {elapsed:.1f} seconds!")
    print(f"📄 Output: {path}")
    print(f"{'='*50}")
    
    return path


if __name__ == "__main__":
    main()
