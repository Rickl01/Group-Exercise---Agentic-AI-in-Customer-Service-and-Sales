# Setup Guide - Customer Service Chatbot

## Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
cd /root/workspace/agentic-chatbot
pip install -r requirements.txt
```

### Step 2: Add Your OpenAI API Key

```bash
# Copy the template
cp config/secrets.example.json config/secrets.json

# Edit the file (use nano, vim, or any editor)
nano config/secrets.json
```

Add your API key:
```json
{
  "OPENAI_API_KEY": "sk-proj-YOUR-KEY-HERE",
  "MODEL_NAME": "gpt-4o-mini"
}
```

### Step 3: Start Jupyter

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

### Step 4: Run the Notebooks

Open and run in this order:

1. **Test Individual Agents** (optional):
   - `01_order_status_agent.ipynb`
   - `02_refund_policy_agent.ipynb`
   - `03_product_suggestion_agent.ipynb`

2. **Run Main Chatbot**:
   - `04_customer_service_chatbot_with_routing.ipynb` ← START HERE

## Troubleshooting

### "Module not found" errors
```bash
pip install --upgrade langgraph langchain langchain-openai
```

### "API key not found"
- Make sure you copied `secrets.example.json` to `secrets.json`
- Check that your API key starts with `sk-`
- Verify the JSON syntax is correct (no trailing commas)

### Jupyter won't start
```bash
pip install jupyter ipykernel
python -m ipykernel install --user
```

### LangGraph errors
```bash
pip install langgraph>=0.2.0 langchain-core>=0.3.0
```

## Running from Command Line (Alternative)

If you prefer Python scripts over notebooks, you can convert:

```bash
# Convert notebook to Python script
jupyter nbconvert --to script 04_customer_service_chatbot_with_routing.ipynb

# Run it
python 04_customer_service_chatbot_with_routing.py
```

## Verification Checklist

Before submitting, verify:

- [ ] All dependencies installed (`pip list | grep langchain`)
- [ ] API key configured (`cat config/secrets.json`)
- [ ] Each notebook runs without errors
- [ ] Memory persists across conversation turns
- [ ] Router correctly identifies query types
- [ ] All three agents respond appropriately

## Next Steps

After setup is complete:
1. Run all test cells in `04_customer_service_chatbot_with_routing.ipynb`
2. Document successful outputs
3. Take screenshots for presentation
4. Review GRADING_CHECKLIST.md for presentation requirements

---

**Having issues?** Check NOTES.md or create an issue in the repository.
