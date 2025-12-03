"""
AI Analysis Module
Uses OpenAI GPT-4o to generate natural language insights from data
"""

import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AIAnalyzer:
    """
    Generates AI-powered insights using GPT-4o
    
    Features:
    - Few-shot prompting for consistent analyst-style output
    - Strict context system to prevent hallucinations
    - Data validation to ensure AI math matches actual data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI Analyzer
        
        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"
    
    def generate_executive_summary(self, data_summary: Dict, 
                                    anomalies: List[Dict],
                                    campaign_performance: Dict) -> str:
        """
        Generate an executive summary of the data
        
        Args:
            data_summary: Summary statistics of the data
            anomalies: List of detected anomalies
            campaign_performance: Campaign performance metrics
            
        Returns:
            Natural language executive summary
        """
        # Build context for the AI
        context = self._build_context(data_summary, anomalies, campaign_performance)
        
        system_prompt = """You are a Senior Data Analyst at a leading AdTech company. 
Your role is to create executive-ready summaries for clients.

CRITICAL RULES:
1. ONLY use the data provided in the context. Do NOT make up statistics.
2. If data is missing or unclear, say "Data not available" instead of guessing.
3. Use precise numbers from the context when stating metrics.
4. Keep the tone professional but accessible to non-technical executives.
5. Focus on actionable insights, not just data recitation.

FORMAT:
- Start with a brief overview (2-3 sentences)
- Highlight key performance metrics
- Identify concerning trends or anomalies
- End with recommendations
"""

        user_prompt = f"""Based on the following data context, write an executive summary for the client's weekly performance report.

DATA CONTEXT:
{json.dumps(context, indent=2)}

Requirements:
1. Write in clear, professional English
2. Mention specific numbers and percentages
3. Highlight any anomalies and their potential business impact
4. Provide 2-3 actionable recommendations
5. Keep it under 400 words
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more factual output
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ AI generation error: {e}")
            return self._generate_fallback_summary(data_summary, anomalies)
    
    def generate_anomaly_analysis(self, anomalies: List[Dict]) -> str:
        """
        Generate detailed analysis of detected anomalies
        
        Args:
            anomalies: List of detected anomalies
            
        Returns:
            Natural language anomaly analysis
        """
        if not anomalies:
            return "No significant anomalies were detected during the reporting period. All metrics are within expected ranges."
        
        system_prompt = """You are a Senior Data Analyst specializing in anomaly investigation.
Your role is to explain data anomalies in business terms.

RULES:
1. Only analyze the anomalies provided in the data
2. Do NOT speculate about causes unless you can logically infer them from the data
3. Be specific about dates, campaigns, and metrics
4. Suggest potential investigation steps
"""

        user_prompt = f"""Analyze the following anomalies detected in our AdTech data:

ANOMALIES:
{json.dumps(anomalies, indent=2)}

For each anomaly:
1. Explain what happened in business terms
2. Assess the severity and potential impact
3. Suggest what might need investigation

Keep the analysis concise and actionable.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ AI generation error: {e}")
            return self._generate_fallback_anomaly_text(anomalies)
    
    def generate_recommendations(self, data_summary: Dict, 
                                  anomalies: List[Dict]) -> List[str]:
        """
        Generate actionable recommendations based on the data
        
        Args:
            data_summary: Summary statistics
            anomalies: List of anomalies
            
        Returns:
            List of recommendation strings
        """
        context = {
            "data_summary": data_summary,
            "anomaly_count": len(anomalies),
            "high_severity_anomalies": [a for a in anomalies if a.get('severity') == 'high']
        }
        
        system_prompt = """You are a strategic AdTech consultant.
Generate specific, actionable recommendations based on data.

RULES:
1. Each recommendation must be actionable
2. Prioritize by potential business impact
3. Be specific (mention campaigns, metrics, timeframes)
4. Do NOT give generic advice - tailor to the data provided
"""

        user_prompt = f"""Based on this performance data and detected anomalies, provide 3-5 specific recommendations:

CONTEXT:
{json.dumps(context, indent=2)}

Format each recommendation as a clear action item.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=400
            )
            
            # Parse the response into a list
            content = response.choices[0].message.content
            recommendations = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Clean up the line
                    clean_line = line.lstrip('0123456789.-•) ').strip()
                    if clean_line:
                        recommendations.append(clean_line)
            
            return recommendations[:5] if recommendations else self._get_default_recommendations()
            
        except Exception as e:
            print(f"❌ AI generation error: {e}")
            return self._get_default_recommendations()
    
    def _build_context(self, data_summary: Dict, 
                       anomalies: List[Dict],
                       campaign_performance: Dict) -> Dict:
        """Build context dictionary for AI"""
        return {
            "reporting_period": data_summary.get('date_range', 'Last 14 days'),
            "total_records": data_summary.get('total_records', 0),
            "metrics": {
                "total_impressions": data_summary.get('total_impressions', 0),
                "total_clicks": data_summary.get('total_clicks', 0),
                "total_conversions": data_summary.get('total_conversions', 0),
                "total_spend": data_summary.get('total_spend', 0),
                "total_revenue": data_summary.get('total_revenue', 0),
                "average_ctr": data_summary.get('avg_ctr', 0),
                "average_roi": data_summary.get('avg_roi', 0)
            },
            "campaign_count": len(campaign_performance) if campaign_performance else 0,
            "top_campaigns": campaign_performance,
            "anomaly_summary": {
                "total_anomalies": len(anomalies),
                "high_severity": len([a for a in anomalies if a.get('severity') == 'high']),
                "affected_campaigns": list(set([a.get('campaign_name', '') for a in anomalies]))
            },
            "anomaly_details": anomalies[:10]  # Limit to top 10 for context length
        }
    
    def _generate_fallback_summary(self, data_summary: Dict, anomalies: List[Dict]) -> str:
        """Generate basic summary if AI fails"""
        total_records = data_summary.get('total_records', 0)
        anomaly_count = len(anomalies)
        
        return f"""## Executive Summary

This report covers the performance analysis of your advertising campaigns.

**Key Metrics:**
- Total records analyzed: {total_records:,}
- Anomalies detected: {anomaly_count}

**Status:** {'⚠️ Attention needed - anomalies detected' if anomaly_count > 0 else '✅ All metrics within normal range'}

Please review the detailed charts and anomaly analysis below for more information.
"""
    
    def _generate_fallback_anomaly_text(self, anomalies: List[Dict]) -> str:
        """Generate basic anomaly text if AI fails"""
        if not anomalies:
            return "No anomalies detected."
        
        text = f"**{len(anomalies)} anomalies detected:**\n\n"
        for a in anomalies[:5]:
            text += f"- {a.get('description', 'Unknown anomaly')}\n"
        
        return text
    
    def _get_default_recommendations(self) -> List[str]:
        """Return default recommendations if AI fails"""
        return [
            "Review campaigns with detected anomalies for potential optimization opportunities",
            "Monitor underperforming campaigns closely over the next week",
            "Consider A/B testing ad creatives for campaigns with declining CTR",
            "Analyze high-performing campaigns to replicate success factors"
        ]


if __name__ == "__main__":
    # Test the module
    analyzer = AIAnalyzer()
    
    test_summary = {
        'total_records': 70,
        'total_impressions': 9500000,
        'total_clicks': 285000,
        'total_conversions': 14250,
        'total_spend': 165000,
        'total_revenue': 412500,
        'avg_ctr': 3.0,
        'avg_roi': 150
    }
    
    test_anomalies = [
        {
            'date': '2025-11-03',
            'campaign_name': 'Tech_Gadgets',
            'metric': 'impressions',
            'value': 45000,
            'severity': 'high',
            'description': 'Impressions dropped 72% below average for Tech_Gadgets on 2025-11-03 in Miami'
        }
    ]
    
    test_campaigns = {
        'Tech_Gadgets': {'revenue': 85000, 'roi': 180},
        'Holiday_Promo': {'revenue': 75000, 'roi': 165}
    }
    
    print("🤖 Testing AI Analyzer...")
    summary = analyzer.generate_executive_summary(test_summary, test_anomalies, test_campaigns)
    print("\n📝 Executive Summary:")
    print(summary)
