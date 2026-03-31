# Ad Optimization Agent — Scaling Note

## Production Deployment Changes

### Observability
- **Logging**: Structure all decisions as JSON (timestamp, channel, old_budget, new_budget, cvr, reasoning)
- **Metrics**: Push conversion/CPA/spend to monitoring system (CloudWatch, Datadog)
- **Alerts**: Trigger if daily shift exceeds ±25% or CPA spikes >20%
- **Dashboard**: Real-time budget allocation view + daily CVR trends per channel

### Retries & Error Handling
- **API failures**: Retry 3x with exponential backoff (1s, 2s, 4s) if data fetch fails
- **Partial data**: If one channel missing (e.g., Social), hold allocation constant, log warning
- **Stale data**: Reject recommendations if data is >6 hours old
- **Budget rounding**: Round allocations to nearest $10 to avoid API errors

### Cost Management
- **Spend caps**: Set daily budget hard limit (e.g., $10K/day); agent cannot exceed
- **LLM cost**: Cache system prompt (GPT: ~80% cost reduction on repeat calls)
- **API rate limits**: Queue budget decisions; run max once per day to avoid quota burn

### Deployment Target
- **Schedule**: Cloud function (AWS Lambda / GCP Cloud Function) triggered daily at 6 AM UTC
- **Data source**: Read from data warehouse (BigQuery, Redshift) or S3 CSVs
- **Decision storage**: Write recommendations to database; human approval before executing
- **Rollback**: Keep last 7 days of allocations; can revert if metric degrades >10%

### Key Reliability Checks
1. Verify no channel allocates <0% or >100%
2. Confirm total budget ≤ daily cap
3. Log all constraints applied
4. Email stakeholders daily summary (allocation + reasoning)

---

**Cost estimate**: ~$20–50/month (function + LLM calls + storage)
