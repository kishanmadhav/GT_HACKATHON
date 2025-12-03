# TrendSpotter: Automated AdTech Insight Engine

A data pipeline that turns raw advertising data into executive-ready PDF or PowerPoint reports. Drop a CSV file, get a professional report with charts and AI-generated insights in about 10 seconds.

## What It Does

Account Managers typically spend 4-6 hours per week pulling together performance reports manually. This tool automates the whole thing:

- Reads advertising/marketing CSV data
- Detects anomalies using machine learning
- Generates charts (impressions, revenue, ROI trends, etc.)
- Creates an AI-written executive summary
- Outputs a formatted PDF or PowerPoint file

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Data Processing | Polars |
| ML | Scikit-Learn (Isolation Forest) |
| AI | OpenAI GPT-4o |
| Charts | Matplotlib |
| PDF | FPDF2 |
| PowerPoint | python-pptx |

## Project Structure

```
GT_HACKATHON/
├── src/
│   ├── main.py              # Full pipeline (original version)
│   ├── data_ingestion.py    # Data loading
│   ├── anomaly_detection.py # Isolation Forest
│   ├── ai_analyzer.py       # GPT-4o integration
│   ├── visualization.py     # Plotly charts
│   ├── report_generator.py  # PDF generation
│   └── file_watcher.py      # Folder monitoring
├── data/
│   ├── input/               # Drop files here
│   └── sample/              # Sample data included
├── output/                  # Generated reports go here
├── screenshots/             # Screenshots of output
├── trendspotter.py          # Main script (use this)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Your API Key

Windows:
```bash
set OPENAI_API_KEY=your_key_here
```

Linux/Mac:
```bash
export OPENAI_API_KEY=your_key_here
```

Or create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

### Run It

Generate a PDF report:
```bash
python trendspotter.py --pdf
```

Generate a PowerPoint report:
```bash
python trendspotter.py --pptx
```

The report will be saved in the `output/` folder.

## Sample Data

The repo includes sample AdTech data at `data/sample/ad_performance.csv` with 70 records covering:

- 5 campaigns (Brand Awareness, Retargeting, Product Launch, etc.)
- 14 days of data
- Metrics: impressions, clicks, conversions, spend, revenue
- Regions: North America, Europe, Asia Pacific, Latin America
- Platforms: Google Ads, Meta Ads, LinkedIn Ads

## What the Report Includes

1. **KPI Summary** - Impressions, clicks, conversions, revenue, spend, ROI
2. **Performance Dashboard** - 4-panel chart with trends
3. **Campaign Breakdown** - Revenue by campaign bar chart
4. **Platform Distribution** - Pie chart of revenue by platform
5. **Executive Summary** - AI-generated analysis of the data
6. **Anomaly Alerts** - Any unusual patterns detected
7. **Recommendations** - AI-generated action items

## Docker

If you prefer containers:

```bash
docker-compose up --build
```

## Input Data Format

Your CSV should have these columns:

| Column | Type | Description |
|--------|------|-------------|
| date | Date | YYYY-MM-DD format |
| campaign_id | String | Unique ID |
| campaign_name | String | Display name |
| impressions | Integer | Ad impressions |
| clicks | Integer | Clicks |
| conversions | Integer | Conversions |
| spend | Float | Ad spend |
| revenue | Float | Revenue generated |
| region | String | Geographic region |
| platform | String | Ad platform |

## Performance

The pipeline runs in about 10 seconds on typical hardware:
- Data loading: under 1s
- Anomaly detection: under 1s
- Chart generation: around 3s
- AI analysis: around 5s
- Report generation: under 1s

## Requirements Met

| Requirement | How |
|-------------|-----|
| Data ingestion (CSV/SQL) | Polars-based loader |
| Data transformation | Metrics calculation, aggregation |
| AI Integration | GPT-4o for summaries |
| PDF/PPTX output | FPDF2 and python-pptx |
| Automated | No manual steps needed |
| Professional output | Charts, tables, styled layout |

## License

MIT

## Author

Built for GT Hackathon 2025
