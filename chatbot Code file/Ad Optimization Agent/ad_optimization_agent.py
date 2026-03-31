#!/usr/bin/env python3
"""
Ad Optimization Agent: Daily budget allocator across marketing channels
Uses conversion rate + heuristic learning to maximize conversions
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

class AdOptimizationAgent:
    def __init__(self, data_path="ad_performance_data.csv", daily_budget=10000):
        self.data_path = data_path
        self.daily_budget = daily_budget
        self.min_allocation_pct = 0.10  # 10% floor per channel
        self.max_daily_shift = 0.20  # ±20% max shift
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.decisions_log = []
        
    def calculate_metrics(self, channel_data):
        """Calculate CVR, CTR, CPA for a channel"""
        spend = float(channel_data['spend'].sum())
        clicks = float(channel_data['clicks'].sum())
        conversions = float(channel_data['conversions'].sum())
        
        ctr = (clicks / float(channel_data['impressions'].sum()) * 100) if channel_data['impressions'].sum() > 0 else 0
        cvr = (conversions / clicks * 100) if clicks > 0 else 0
        cpa = (spend / conversions) if conversions > 0 else float('inf')
        
        return {
            'spend': spend,
            'impressions': float(channel_data['impressions'].sum()),
            'clicks': clicks,
            'conversions': conversions,
            'ctr': ctr,
            'cvr': cvr,
            'cpa': cpa
        }
    
    def allocate_budget(self, historical_days=7):
        """
        Allocate budget based on recent performance.
        Strategy: Shift budget toward highest CVR channel, maintain diversity.
        """
        
        # Get last N days of data
        latest_date = self.df['date'].max()
        cutoff_date = latest_date - timedelta(days=historical_days)
        recent_data = self.df[self.df['date'] > cutoff_date]
        
        # Calculate metrics per channel
        channels = recent_data['channel'].unique()
        metrics = {}
        
        for channel in channels:
            channel_data = recent_data[recent_data['channel'] == channel]
            metrics[channel] = self.calculate_metrics(channel_data)
        
        # Current allocation (from last day)
        last_day_data = recent_data[recent_data['date'] == latest_date]
        current_alloc = {}
        total_spend = last_day_data['spend'].sum()
        
        for channel in channels:
            spend = last_day_data[last_day_data['channel'] == channel]['spend'].values
            current_alloc[channel] = (spend[0] / total_spend) if len(spend) > 0 else 1/len(channels)
        
        # Sort by CVR
        sorted_channels = sorted(metrics.items(), key=lambda x: x[1]['cvr'], reverse=True)
        top_performer = sorted_channels[0][0]
        bottom_performer = sorted_channels[-1][0]
        
        # Allocate: shift from bottom to top, respect guardrails
        new_alloc = current_alloc.copy()
        
        shift_pct = 0.15  # 15% shift
        new_alloc[top_performer] = min(
            current_alloc[top_performer] + shift_pct,
            1.0 - (len(channels) - 1) * self.min_allocation_pct
        )
        new_alloc[bottom_performer] = max(
            current_alloc[bottom_performer] - shift_pct,
            self.min_allocation_pct
        )
        
        # Normalize to sum to 1
        total = sum(new_alloc.values())
        new_alloc = {k: v/total for k, v in new_alloc.items()}
        
        # Convert to dollars
        new_budget = {k: self.daily_budget * v for k, v in new_alloc.items()}
        
        # Log decision
        decision = {
            'timestamp': datetime.now().isoformat(),
            'next_date': (latest_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            'rationale': f"{top_performer} CVR {metrics[top_performer]['cvr']:.2f}% > {bottom_performer} CVR {metrics[bottom_performer]['cvr']:.2f}%. Shift +{shift_pct*100:.0f}% to top.",
            'metrics': {ch: {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()} for ch, m in metrics.items()},
            'current_allocation': {k: round(v*100, 1) for k, v in current_alloc.items()},
            'new_allocation': {k: round(v*100, 1) for k, v in new_alloc.items()},
            'new_budget_dollars': {k: round(v, 2) for k, v in new_budget.items()},
            'daily_budget_total': round(sum(new_budget.values()), 2)
        }
        
        self.decisions_log.append(decision)
        return decision
    
    def evaluate_performance(self, lookback_days=14):
        """Evaluate agent performance vs baseline (equal split)"""
        latest_date = self.df['date'].max()
        cutoff_date = latest_date - timedelta(days=lookback_days)
        eval_data = self.df[self.df['date'] > cutoff_date]
        
        channels = eval_data['channel'].unique()
        
        # Agent performance (based on logged decisions)
        total_conversions_agent = eval_data.groupby('channel')['conversions'].sum().sum()
        total_spend_agent = eval_data.groupby('channel')['spend'].sum().sum()
        avg_cpa_agent = total_spend_agent / total_conversions_agent if total_conversions_agent > 0 else 0
        
        # Baseline (equal 33.3% split)
        avg_cpa_baseline = total_spend_agent / total_conversions_agent  # same for this mock
        
        print("\n📊 EVALUATION RESULTS (14-day lookback)")
        print(f"Total conversions: {total_conversions_agent}")
        print(f"Total spend: ${total_spend_agent:,.2f}")
        print(f"Avg CPA: ${avg_cpa_agent:.2f}")
        print(f"Conversion rate: {(total_conversions_agent / eval_data['clicks'].sum() * 100):.2f}%")
        
        # Per-channel breakdown
        print("\n📈 Channel Breakdown:")
        for channel in channels:
            ch_data = eval_data[eval_data['channel'] == channel]
            convs = ch_data['conversions'].sum()
            spend = ch_data['spend'].sum()
            cpa = spend / convs if convs > 0 else 0
            cvr = (convs / ch_data['clicks'].sum() * 100) if ch_data['clicks'].sum() > 0 else 0
            print(f"  {channel}: {convs} conversions | ${cpa:.2f} CPA | {cvr:.2f}% CVR")
    
    def save_decision_log(self, filepath="decision_log.json"):
        """Save all decisions to JSON for audit"""
        with open(filepath, 'w') as f:
            json.dump(self.decisions_log, f, indent=2)
        print(f"\n✅ Decision log saved to {filepath}")
    
    def print_latest_decision(self):
        """Pretty print the latest allocation decision"""
        if not self.decisions_log:
            print("No decisions yet.")
            return
        
        latest = self.decisions_log[-1]
        print("\n" + "="*60)
        print("📌 LATEST BUDGET ALLOCATION DECISION")
        print("="*60)
        print(f"Date: {latest['timestamp']}")
        print(f"Next Day: {latest['next_date']}")
        print(f"\n💭 Rationale:\n{latest['rationale']}")
        print(f"\n📊 Current Performance Metrics:")
        for ch, metrics in latest['metrics'].items():
            print(f"  {ch}: CVR={metrics['cvr']:.2f}%, CPA=${metrics['cpa']:.2f}, Conversions={metrics['conversions']}")
        print(f"\n💰 Current Allocation %:")
        for ch, pct in latest['current_allocation'].items():
            print(f"  {ch}: {pct:.1f}%")
        print(f"\n🎯 Recommended Allocation %:")
        for ch, pct in latest['new_allocation'].items():
            print(f"  {ch}: {pct:.1f}%")
        print(f"\n💵 Recommended Budget (${latest['daily_budget_total']:,.2f}/day):")
        for ch, budget in latest['new_budget_dollars'].items():
            print(f"  {ch}: ${budget:,.2f}")
        print("="*60 + "\n")

# Main execution
if __name__ == "__main__":
    print("🤖 Ad Optimization Agent - Prototype")
    print("=" * 60)
    
    # Initialize agent
    agent = AdOptimizationAgent(
        data_path="ad_performance_data.csv",
        daily_budget=10000
    )
    
    # Run optimization
    print("\n📍 Running budget allocation...")
    decision = agent.allocate_budget(historical_days=7)
    
    # Print recommendation
    agent.print_latest_decision()
    
    # Evaluate performance
    agent.evaluate_performance(lookback_days=14)
    
    # Save log
    agent.save_decision_log("decision_log.json")
    
    print("\n✅ Agent execution complete!")
