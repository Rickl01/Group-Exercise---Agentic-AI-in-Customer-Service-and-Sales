# Ad Optimization Agent — Design Doc

## Agent Role
Autonomous budget allocator that daily rebalances ad spend across channels (Search, Social, Display) to maximize conversions. Uses conversion rate + heuristic exploration to shift budget toward high-performers while maintaining channel diversity.

## Inputs
- **Daily performance data** (CSV): date, channel, spend, impressions, clicks, conversions
- **Constraints**: min/max budget, max daily shift (±20%), minimum allocation floor (10% per channel)
- **Target metric**: maximize conversions OR minimize CPA (cost per acquisition)

## Outputs
- **Budget allocation** (next day): {Search: $X, Social: $Y, Display: $Z}
- **Rationale log**: decision reason (e.g., "Social +15% due to CVR 3.2% vs 2.1%")
- **Performance metrics**: total conversions, avg CPA, channel breakdown

## Rules & Guardrails

| Rule | Details |
|------|---------|
| **Privacy** | No PII in logs; anonymize impressions/clicks |
| **Brand Tone** | Maintain minimum allocation (≥10%) per channel for 30+ days to prevent brand dilution |
| **Budget Cap** | Daily shift ±20% max; never reallocate >50% in single day |
| **Floor Policy** | If channel drops to 0% for >2 days, auto-restore to 10% |
| **Logging** | JSON log each decision with timestamp, metric, rationale |
| **Audit Trail** | Store all recommendations (approved/rejected) for 90 days |

## Evaluation Metric
- **Primary**: Total conversions (higher = better)
- **Secondary**: Average CPA (lower = better)
- **Tertiary**: Budget utilization rate (% of daily budget spent)
- **Baseline**: Equal 33.3% split across channels for comparison

## Agent Reasoning Loop
1. Read yesterday's performance data
2. Calculate CVR, CPA, CTR per channel
3. Identify top performer (highest CVR)
4. Shift +10–20% from bottom performer → top performer
5. Apply guardrails (floor, cap, daily limit)
6. Log decision + rationale
7. Output recommended allocation for today

---

**Version**: 1.0 | **Last Updated**: Mar 30, 2026 | **Status**: Ready for prototyping
