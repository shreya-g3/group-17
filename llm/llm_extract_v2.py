"""
USAGE:
  python llm_extract_v2.py --mode zero_shot --limit 10
  python llm_extract_v2.py --mode few_shot  --limit 10
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
DATA_ROOT = Path("../ebm_nlp_2_00")
DOCUMENTS_DIR = DATA_ROOT / "documents"
TEST_IDS_FILE = DATA_ROOT / "annotations" / "aggregated" / "starting_spans" / "participants" / "test" / "gold"

OUTPUT_DIR = Path("llm_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0

PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60


# ============================================================
# System prompt
# ============================================================
SYSTEM_PROMPT = """You are an expert at analysing clinical trial abstracts and annotating them using the EBM-NLP convention.

Your task: Extract PICO elements from clinical abstracts, following the specific EBM-NLP annotation guidelines described below.

=====================================================
PICO elements to extract:
=====================================================

### PARTICIPANTS (P) - IMPORTANT: Use the BROAD EBM-NLP definition
Under EBM-NLP conventions, "Participants" includes ALL of the following when they appear in the abstract:
1. The people being studied (e.g. "200 adults", "children aged 6-12", "patients")
2. The medical condition(s) they have (e.g. "type 2 diabetes", "asthma", "seasonal allergic rhinitis")
3. Demographic or geographic context (e.g. "Ontario, Quebec and Manitoba")
4. Abbreviations of the condition wherever they appear (e.g. "SAR", "COPD", "HTN")
5. Sample size mentions (e.g. "243 randomized", "n=150")

Extract ALL occurrences - if "SAR" appears three times in the abstract, extract it three times as separate spans.

### INTERVENTIONS (I)
Treatments, drugs, procedures, or therapies given to participants, including:
- Drug names with doses (e.g. "metformin 500mg daily")
- Placebos and comparators (e.g. "matching placebo", "standard care")
- Procedures and therapies (e.g. "weekly physiotherapy sessions")

### OUTCOMES (O)
What the study measured, including:
- Primary and secondary outcome measures
- Biomarkers, lab values, scores (e.g. "HbA1c", "pain intensity")
- Clinical endpoints (e.g. "hospitalisation rate", "mortality")

DO NOT include in outcomes:
- Commentary about the results (e.g. "X was better than Y")
- Adverse event descriptions (extract just the event type, not the narrative)

=====================================================
RULES:
=====================================================
1. Extract spans EXACTLY as they appear in the text (verbatim, preserving case and punctuation).
2. Extract ALL mentions - including repeated abbreviations and condition names throughout the text.
3. Use short, focused spans. Prefer "seasonal allergic rhinitis" over "patients with seasonal allergic rhinitis".
4. If a category has no content, return an empty list.
5. Output JSON ONLY, no explanations.

Respond in this EXACT JSON format:
{
  "participants": ["span1", "span2", ...],
  "interventions": ["span1", "span2", ...],
  "outcomes": ["span1", "span2", ...]
}
"""

# ============================================================
# FEW-SHOT EXAMPLES - EBM-NLP-style broad annotations
# ============================================================
FEW_SHOT_EXAMPLES = [
    {
        "abstract": (
            "Comparison of budesonide Turbuhaler with budesonide aqua in the treatment of "
            "seasonal allergic rhinitis. OBJECTIVE To compare the effect of budesonide "
            "Turbuhaler 400 microg/day with budesonide aqua 256 microg/day in the treatment "
            "of seasonal allergic rhinitis (SAR). Two hundred and eighty-four out-patients "
            "with SAR in Ontario, Quebec and Manitoba were enrolled; 243 randomized. "
            "Primary outcome was mean daily nasal symptom scores. Secondary outcomes "
            "included eye symptoms and quality of life. Adverse events were monitored."
        ),
        "answer": {
            "participants": [
                "seasonal allergic rhinitis",
                "seasonal allergic rhinitis",
                "SAR",
                "Two hundred and eighty-four out-patients with SAR",
                "Ontario, Quebec and Manitoba",
                "243 randomized",
                "SAR"
            ],
            "interventions": [
                "budesonide Turbuhaler",
                "budesonide aqua",
                "budesonide Turbuhaler 400 microg/day",
                "budesonide aqua 256 microg/day"
            ],
            "outcomes": [
                "mean daily nasal symptom scores",
                "eye symptoms",
                "quality of life",
                "Adverse events"
            ]
        }
    },
    {
        "abstract": (
            "A randomized controlled trial of metformin in 300 adults with newly diagnosed "
            "type 2 diabetes mellitus (T2DM). Participants received metformin 500mg twice daily "
            "or matching placebo for 24 weeks. The primary endpoint was change in HbA1c. "
            "Secondary endpoints included fasting glucose, body weight, and blood pressure."
        ),
        "answer": {
            "participants": [
                "300 adults with newly diagnosed type 2 diabetes mellitus",
                "type 2 diabetes mellitus",
                "T2DM"
            ],
            "interventions": [
                "metformin",
                "metformin 500mg twice daily",
                "matching placebo"
            ],
            "outcomes": [
                "HbA1c",
                "fasting glucose",
                "body weight",
                "blood pressure"
            ]
        }
    },
    {
        "abstract": (
            "This multicentre study enrolled 180 patients with chronic obstructive pulmonary "
            "disease (COPD). Patients were randomised to inhaled tiotropium 18 microg daily "
            "or placebo for 12 months. Primary outcome was forced expiratory volume (FEV1). "
            "Secondary outcomes were COPD exacerbations and quality of life using the SGRQ."
        ),
        "answer": {
            "participants": [
                "180 patients with chronic obstructive pulmonary disease",
                "chronic obstructive pulmonary disease",
                "COPD",
                "COPD"
            ],
            "interventions": [
                "inhaled tiotropium",
                "inhaled tiotropium 18 microg daily",
                "placebo"
            ],
            "outcomes": [
                "forced expiratory volume",
                "FEV1",
                "COPD exacerbations",
                "quality of life",
                "SGRQ"
            ]
        }
    }
]


# ============================================================
# Helper function
# ============================================================

def load_test_document_ids(limit=None):
    if not TEST_IDS_FILE.exists():
        raise FileNotFoundError(f"Test gold directory not found at: {TEST_IDS_FILE}")

    doc_ids = []
    for filename in sorted(os.listdir(TEST_IDS_FILE)):
        if filename.endswith(".AGGREGATED.ann"):
            doc_id = filename.split(".")[0]
            doc_ids.append(doc_id)

    if limit is not None:
        doc_ids = doc_ids[:limit]

    return doc_ids


def load_abstract(doc_id):
    text_file = DOCUMENTS_DIR / f"{doc_id}.txt"
    if not text_file.exists():
        return None
    with open(text_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_messages(abstract, mode):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if mode == "few_shot":
        for example in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Abstract:\n{example['abstract']}"
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(example["answer"], indent=2)
            })

    messages.append({
        "role": "user",
        "content": f"Abstract:\n{abstract}"
    })

    return messages


def extract_pico_from_abstract(client, abstract, mode):
    messages = build_messages(abstract, mode)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        extracted = json.loads(content)

        for key in ["participants", "interventions", "outcomes"]:
            if key not in extracted:
                extracted[key] = []

        return extracted, response.usage.prompt_tokens, response.usage.completion_tokens

    except json.JSONDecodeError as e:
        print(f"  [warning] JSON parse error: {e}")
        return None, 0, 0
    except Exception as e:
        print(f"  [error] API call failed: {e}")
        return None, 0, 0


def calculate_cost(input_tokens, output_tokens):
    return (
        input_tokens * PRICE_INPUT_PER_1M / 1_000_000
        + output_tokens * PRICE_OUTPUT_PER_1M / 1_000_000
    )


def main():
    parser = argparse.ArgumentParser(description="LLM PICO extraction v2 (EBM-NLP-aligned)")
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "few_shot"])
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        print("ERROR: OPENAI_API_KEY not found in .env file!")
        return

    client = OpenAI(api_key=api_key)

    print(f"Loading up to {args.limit} test document IDs...")
    doc_ids = load_test_document_ids(limit=args.limit)
    print(f"Found {len(doc_ids)} documents.\n")

    print(f"Mode:          {args.mode}")
    print(f"Model:         {MODEL}")
    print(f"Documents:     {len(doc_ids)}")


    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    failures = 0

    for doc_id in tqdm(doc_ids, desc=f"Extracting v2 ({args.mode})"):
        abstract = load_abstract(doc_id)
        if abstract is None:
            failures += 1
            continue

        extracted, in_tokens, out_tokens = extract_pico_from_abstract(
            client, abstract, args.mode
        )

        if extracted is None:
            failures += 1
            continue

        total_input_tokens += in_tokens
        total_output_tokens += out_tokens

        results.append({
            "doc_id": doc_id,
            "abstract": abstract,
            "extracted": extracted,
        })

    # Save Results
    output_file = OUTPUT_DIR / f"extracted_v2_{args.mode}_{len(doc_ids)}docs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    actual_cost = calculate_cost(total_input_tokens, total_output_tokens)
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE (v2)")
    print("=" * 60)
    print(f"Successful: {len(results)} / {len(doc_ids)}")
    print(f"Failures:   {failures}")
    print(f"Input tokens:  {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Actual cost:   ${actual_cost:.4f}")
    print(f"Results saved: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()