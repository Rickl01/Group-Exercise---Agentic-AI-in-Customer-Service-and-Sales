#!/usr/bin/env python3
"""
Ad Optimization Agent: OpenAI-assisted daily budget allocator across marketing channels.

What changed:
- Keeps your original rule-based guardrails
- Optionally asks OpenAI for a recommendation and rationale
- Falls back to heuristic allocation if OPENAI_API_KEY is not set
- Never hardcodes the API key in this file
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class AdOptimizationAgent:
    def __init__(self, data_path="ad_performance_data.csv", daily_budget=10000):
        self.data_path = data_path
        self.daily_budget = daily_budget
        self.min_allocation_pct = 0.10  # 10% floor per channel
        self.max_daily_shift = 0.20     # ±20% max shift
        self.df = pd.read_csv(data_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.decisions_log = []

    def calculate_metrics(self, channel_data):
        """Calculate CVR, CTR, CPA for a channel."""
        spend = float(channel_data["spend"].sum())
        clicks = float(channel_data["clicks"].sum())
        conversions = float(channel_data["conversions"].sum())
        impressions = float(channel_data["impressions"].sum())

        ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
        cvr = (conversions / clicks * 100) if clicks > 0 else 0.0
        cpa = (spend / conversions) if conversions > 0 else float("inf")

        return {
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "ctr": ctr,
            "cvr": cvr,
            "cpa": cpa,
        }

    def get_recent_metrics(self, historical_days=7):
        """Collect recent metrics and current allocation from the latest day."""
        latest_date = self.df["date"].max()
        cutoff_date = latest_date - timedelta(days=historical_days)
        recent_data = self.df[self.df["date"] > cutoff_date]

        channels = list(recent_data["channel"].unique())
        metrics = {}
        for channel in channels:
            channel_data = recent_data[recent_data["channel"] == channel]
            metrics[channel] = self.calculate_metrics(channel_data)

        last_day_data = recent_data[recent_data["date"] == latest_date]
        total_spend = float(last_day_data["spend"].sum())

        current_alloc = {}
        for channel in channels:
            spend = last_day_data[last_day_data["channel"] == channel]["spend"].values
            current_alloc[channel] = (float(spend[0]) / total_spend) if len(spend) > 0 and total_spend > 0 else 1 / len(channels)

        return latest_date, channels, metrics, current_alloc

    def heuristic_allocation(self, metrics, current_alloc, channels):
        """Original rule-based allocation: move 15% from worst CVR to best CVR."""
        sorted_channels = sorted(metrics.items(), key=lambda x: x[1]["cvr"], reverse=True)
        top_performer = sorted_channels[0][0]
        bottom_performer = sorted_channels[-1][0]

        new_alloc = current_alloc.copy()
        shift_pct = 0.15

        max_top_allowed = 1.0 - (len(channels) - 1) * self.min_allocation_pct
        new_alloc[top_performer] = min(current_alloc[top_performer] + shift_pct, max_top_allowed)
        new_alloc[bottom_performer] = max(current_alloc[bottom_performer] - shift_pct, self.min_allocation_pct)

        total = sum(new_alloc.values())
        new_alloc = {k: v / total for k, v in new_alloc.items()}

        rationale = (
            f"{top_performer} CVR {metrics[top_performer]['cvr']:.2f}% > "
            f"{bottom_performer} CVR {metrics[bottom_performer]['cvr']:.2f}%. "
            f"Shift +{shift_pct * 100:.0f}% to top performer."
        )

        return new_alloc, rationale, "heuristic"

    def ai_allocation(self, metrics, current_alloc, channels):
        """
        Ask OpenAI for a recommendation.
        Returns (allocation_dict, rationale, source) or None if unavailable/invalid.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return None

        try:
            client = OpenAI(api_key=api_key)

            prompt = f"""
You are helping allocate a daily advertising budget across channels.

Rules you MUST follow:
- Output valid JSON only
- Percentages must sum to 100
- No channel below {self.min_allocation_pct * 100:.0f}%
- Do not recommend more than {self.max_daily_shift * 100:.0f}% absolute change for any single channel vs current allocation
- Favor higher conversion rate (CVR), but consider CPA too

Current allocation percentages:
{json.dumps({k: round(v * 100, 2) for k, v in current_alloc.items()}, indent=2)}

Channel metrics:
{json.dumps(metrics, indent=2)}

Return this exact shape:
{{
  "allocation_pct": {{
    "Search": number,
    "Social": number,
    "Display": number
  }},
  "rationale": "short explanation"
}}
""".strip()

            response = client.responses.create(
                model="gpt-5",
                input=prompt
            )

            text = response.output_text.strip()
            payload = json.loads(text)

            alloc_pct = payload["allocation_pct"]
            rationale = payload["rationale"]

            # Convert % to decimal
            new_alloc = {k: float(v) / 100.0 for k, v in alloc_pct.items()}

            # Validate channels
            if set(new_alloc.keys()) != set(channels):
                return None

            # Validate floors and daily change
            for ch in channels:
                if new_alloc[ch] < self.min_allocation_pct:
                    return None
                if abs(new_alloc[ch] - current_alloc[ch]) > self.max_daily_shift + 1e-9:
                    return None

            total = sum(new_alloc.values())
            if total <= 0:
                return None

            # Normalize tiny rounding drift
            new_alloc = {k: v / total for k, v in new_alloc.items()}

            return new_alloc, rationale, "openai"

        except Exception:
            return None

    def allocate_budget(self, historical_days=7):
        """Allocate budget with OpenAI first, then heuristic fallback."""
        latest_date, channels, metrics, current_alloc = self.get_recent_metrics(historical_days=historical_days)

        ai_result = self.ai_allocation(metrics, current_alloc, channels)
        if ai_result is not None:
            new_alloc, rationale, decision_source = ai_result
        else:
            new_alloc, rationale, decision_source = self.heuristic_allocation(metrics, current_alloc, channels)

        new_budget = {k: self.daily_budget * v for k, v in new_alloc.items()}

        decision = {
            "timestamp": datetime.now().isoformat(),
            "next_date": (latest_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "decision_source": decision_source,
            "rationale": rationale,
            "metrics": {
                ch: {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()}
                for ch, m in metrics.items()
            },
            "current_allocation": {k: round(v * 100, 1) for k, v in current_alloc.items()},
            "new_allocation": {k: round(v * 100, 1) for k, v in new_alloc.items()},
            "new_budget_dollars": {k: round(v, 2) for k, v in new_budget.items()},
            "daily_budget_total": round(sum(new_budget.values()), 2),
        }

        self.decisions_log.append(decision)
        return decision

    def save_decision_log(self, filepath="decision_log_openai.json"):
        """Save all decisions to JSON for audit."""
        with open(filepath, "w") as f:
            json.dump(self.decisions_log, f, indent=2)
        print(f"\nSaved decision log to {filepath}")

    def print_latest_decision(self):
        """Pretty print the latest allocation decision."""
        if not self.decisions_log:
            print("No decisions yet.")
            return

        latest = self.decisions_log[-1]
        print("\n" + "=" * 60)
        print("LATEST BUDGET ALLOCATION DECISION")
        print("=" * 60)
        print(f"Date: {latest['timestamp']}")
        print(f"Next Day: {latest['next_date']}")
        print(f"Decision Source: {latest['decision_source']}")
        print(f"\nRationale:\n{latest['rationale']}")
        print("\nCurrent Performance Metrics:")
        for ch, metrics in latest["metrics"].items():
            print(
                f"  {ch}: CVR={metrics['cvr']:.2f}%, "
                f"CPA=${metrics['cpa']:.2f}, "
                f"Conversions={metrics['conversions']}"
            )
        print("\nCurrent Allocation %:")
        for ch, pct in latest["current_allocation"].items():
            print(f"  {ch}: {pct:.1f}%")
        print("\nRecommended Allocation %:")
        for ch, pct in latest["new_allocation"].items():
            print(f"  {ch}: {pct:.1f}%")
        print(f"\nRecommended Budget (${latest['daily_budget_total']:,.2f}/day):")
        for ch, budget in latest["new_budget_dollars"].items():
            print(f"  {ch}: ${budget:,.2f}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Ad Optimization Agent - OpenAI Assisted")
    print("=" * 60)

    print("\nBefore running, set your key:")
    print('  export OPENAI_API_KEY="your_real_key_here"')
    print("\nIf no key is set, the script falls back to the built-in heuristic.\n")

    agent = AdOptimizationAgent(
        data_path="ad_performance_data.csv",
        daily_budget=10000
    )

    print("Running budget allocation...")
    agent.allocate_budget(historical_days=7)
    agent.print_latest_decision()
    agent.save_decision_log("decision_log_openai.json")
