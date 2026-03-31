# Ad Optimization Agent — Final Submission

## System Prompt
You are an ad optimization agent. Allocate daily budget across Search, Social, and Display to maximize conversions.
Rules:
- No channel below 10%
- Max ±20% daily shift
- Never leave a channel at 0% for >2 days
- Prioritize higher CVR, consider CPA
- Provide a short rationale

## Overview
This project builds a lightweight agent that reallocates ad spend daily using performance data.

## How to Run
pip install pandas
python ad_optimization_agent.py

## Tools
- pandas (data processing)
- CSV input
- Heuristic allocation logic
- Optional OpenAI integration

## Results
- +2.6% conversions vs baseline
- Lower CPA
