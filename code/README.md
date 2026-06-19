# Damage Claim Verification System

```bash
pip install -r code/requirements.txt
export GEMINI_API_KEY=your_key_here      # Windows PowerShell: $env:GEMINI_API_KEY="your_key_here"

# Run on test claims (Strategy B recommended)
python code/main.py --strategy B

# Evaluate on sample claims (both strategies)
python code/evaluation/main.py --strategy both
```
