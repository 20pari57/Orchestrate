# 🔍 Damage Claim Verification System

> AI-powered multi-modal damage claim verifier built for the **HackerRank Orchestrate 24-hour Hackathon**.  
> Analyzes images + chat transcripts to decide if a damage claim is `supported`, `contradicted`, or `not_enough_information`.

---

## 🧠 How It Works

```
claims.csv + images
       │
       ▼
┌─────────────────────────────────┐
│   Load claim + user history     │
│   + evidence requirements       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Encode images as base64        │
│  Build structured prompt        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Gemini Vision API             │
│   Strategy A: single-shot       │
│   Strategy B: chain-of-thought  │
└────────────┬────────────────────┘
             │
             ▼
        output.csv
```

---

## 📁 Project Structure

```
.
├── code/
│   ├── main.py                  # Main pipeline entry point
│   └── evaluation/
│       ├── main.py              # Evaluation script
│       └── evaluation_report.md # Strategy comparison + metrics
├── dataset/
│   ├── claims.csv               # Test claims (input only)
│   ├── sample_claims.csv        # Labeled examples for evaluation
│   ├── user_history.csv         # User risk history
│   ├── evidence_requirements.csv
│   └── images/
│       ├── sample/
│       └── test/
├── output.csv                   # Final predictions
├── eval_output_A.csv            # Strategy A evaluation results
├── eval_output_B.csv            # Strategy B evaluation results
└── pipeline.log                 # Full execution log
```

---

## ⚡ Quickstart

### 1. Install dependencies

```bash
pip install google-generativeai pandas tqdm Pillow python-dotenv
```

### 2. Set your API key

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"

# Mac / Linux
export GEMINI_API_KEY="your_key_here"
```

Get a free key at 👉 [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 3. Run the pipeline

```bash
# Strategy A — single-shot (faster)
python code/main.py --strategy A

# Strategy B — chain-of-thought (more accurate) ✅ recommended
python code/main.py --strategy B

# Test on first 5 rows only
python code/main.py --strategy B --limit 5
```

### 4. Run evaluation

```bash
python code/evaluation/main.py --strategy B
```

---

## 🎯 Output Schema

Each row in `output.csv` contains:

| Column | Description |
|--------|-------------|
| `claim_status` | `supported` / `contradicted` / `not_enough_information` |
| `severity` | `none` / `low` / `medium` / `high` / `unknown` |
| `issue_type` | `dent`, `crack`, `scratch`, `water_damage`, etc. |
| `object_part` | Affected part of car / laptop / package |
| `evidence_standard_met` | `true` / `false` |
| `risk_flags` | Semicolon-separated flags or `none` |
| `valid_image` | `true` / `false` |
| `supporting_image_ids` | Image IDs that support the decision |
| `claim_status_justification` | Short image-grounded explanation |
| `evidence_standard_met_reason` | Why evidence was or wasn't sufficient |

---

## 🔬 Strategies

### Strategy A — Single-Shot
One API call per claim. All images + context sent together.  
**Faster and cheaper** — good for large batches.

### Strategy B — Chain-of-Thought ✅
Two API calls per claim:
1. **Describe** — model objectively describes what it sees in each image
2. **Decide** — model uses the description + context to produce the final verdict

**More accurate** — especially for edge cases and mismatch detection.

---

## 📊 Results

| Metric | Strategy A | Strategy B |
|--------|-----------|-----------|
| `claim_status` accuracy | 65.0% | 80.0% |
| `severity` accuracy | 60.0% | 75.0% |
| `issue_type` accuracy | 70.0% | 80.0% |
| `evidence_standard_met` accuracy | 75.0% | 85.0% |
| `valid_image` accuracy | 80.0% | 90.0% |
| Estimated cost (20 claims) | $0.17 | $0.26 |
| Runtime | 125.82s | 114.66s |

> Final `output.csv` was generated using **Strategy B**.

---

## 🛡️ Reliability Features

- ✅ Retry with exponential backoff (up to 5 retries, max 32s delay)
- ✅ Partial saves every 5 rows → `output_partial.csv`
- ✅ Concurrent processing (3 workers via ThreadPoolExecutor)
- ✅ Graceful fallback to `not_enough_information` on errors
- ✅ Prompt injection detection (`text_instruction_present` risk flag)

---

## 🤖 Model

**Gemini 2.0 Flash** (`gemini-2.0-flash`)  
Free tier: 1,500 requests/day — no credit card required.

---

## 📋 Submission Checklist

- [x] `output.csv` — 44 rows, 14 columns
- [x] `code.zip` — full runnable solution
- [x] `log.txt` — AI chat transcript
- [x] `evaluation_report.md` — strategy comparison + metrics
