# TrendSpotter: Automated AdTech Insight Engine

A data pipeline that turns raw advertising data into executive-ready PDF or PowerPoint reports. Drop a CSV file, get a professional report with charts and AI-generated insights in about 10 seconds.

## What It Does

Account Managers typically spend 4-6 hours per week pulling together performance reports manually. This tool automates the whole thing:

- Reads any CSV, JSON, or Parquet file (columns are auto-detected)
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

Generate a PDF report (uses sample data by default):
```bash
python trendspotter.py --pdf
```

Generate a PowerPoint report:
```bash
python trendspotter.py --pptx
```

Use your own data file:
```bash
python trendspotter.py your_data.csv --pdf
python trendspotter.py export.json --pptx
python trendspotter.py database.sqlite --pdf
```

The report will be saved in the `output/` folder.

## Sample Data

The repo includes multiple sample data files in different formats at `data/sample/`:

| File | Format | Records | Description |
|------|--------|---------|-------------|
| `ad_performance.csv` | CSV | 70 | Standard AdTech data with all fields |
| `marketing_export.csv` | CSV | 12 | Different column names (impr, clk, conv) |
| `ecommerce_data.json` | JSON | 12 | E-commerce format (views, visits, orders) |
| `analytics_export.ndjson` | NDJSON | 10 | Line-delimited JSON (shows, hits, actions) |
| `campaign_db.sqlite` | SQLite | 10 | Database format with campaigns table |

Each file uses different column naming conventions to demonstrate the auto-detection feature.

### Testing Different Formats

```bash
# Standard CSV
python trendspotter.py data/sample/ad_performance.csv --pdf

# CSV with different column names
python trendspotter.py data/sample/marketing_export.csv --pdf

# JSON array
python trendspotter.py data/sample/ecommerce_data.json --pdf

# Newline-delimited JSON
python trendspotter.py data/sample/analytics_export.ndjson --pdf

# SQLite database
python trendspotter.py data/sample/campaign_db.sqlite --pdf
```

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

The pipeline accepts **any CSV, JSON, or Parquet file**. It automatically detects and maps your columns to the expected fields using pattern matching. No need to rename your columns.

### Auto-detected Column Mappings

| Your Column Names | Maps To |
|-------------------|---------|
| date, timestamp, day, created, period | date |
| campaign_name, name, title, ad_name, ad_group | campaign_name |
| impressions, views, impr, shows, reach, display | impressions |
| clicks, visits, sessions, hits, taps | clicks |
| conversions, orders, purchases, leads, signups | conversions |
| spend, cost, expense, budget, ad_cost | spend |
| revenue, income, sales, value, earnings, amount | revenue |
| region, country, location, geo, market, area | region |
| platform, source, channel, network, medium | platform |

If a column can't be mapped, the pipeline creates a synthetic one with default values. This means you can throw almost any data file at it and get a report.

### Example: Different Data Formats

Standard AdTech format:
```csv
date,campaign_name,impressions,clicks,spend,revenue
2024-01-01,Winter Sale,50000,1200,500,1500
```

E-commerce format:
```csv
timestamp,ad_group,views,visits,cost,sales_value,country
2024-01-01,Promo A,50000,1200,500,1500,USA
```

Analytics export:
```json
[{"day": "2024-01-01", "name": "Campaign X", "shows": 50000, "hits": 1200}]
```

All of these work without any changes.

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
| Data ingestion (CSV/JSON/DB) | Polars-based loader with auto-detection |
| Data transformation | Auto column mapping, metrics calculation |
| AI Integration | GPT-4o for summaries |
| PDF/PPTX output | FPDF2 and python-pptx |
| Automated | No manual steps needed |
| Professional output | Charts, tables, styled layout |

## License

MIT

## Author

Built for GT Hackathon 2025
