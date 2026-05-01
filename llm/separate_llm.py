import os
import json
import glob
from pathlib import Path

from dataloader_utils import get_doc_ids, get_all


LLM_OUTPUT_DIR = Path("llm_outputs")


# ============================================================
# Cache management
# ============================================================

def _find_any_cache(mode):
    """Find any cached extraction file for this mode (regardless of doc count)."""
    pattern = str(LLM_OUTPUT_DIR / f"extracted_v2_{mode}_*docs.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None  # latest/largest


def _load_cached_extractions_by_docids(mode, required_doc_ids):
    """
    Load cached extractions, keyed by doc_id.
    Returns a dict {doc_id: extraction_record}.
    """
    cache_file = _find_any_cache(mode)
    if cache_file is None:
        return {}, None

    with open(cache_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    cache_by_id = {rec["doc_id"]: rec for rec in records}

    covered = sum(1 for d in required_doc_ids if d in cache_by_id)
    missing = len(required_doc_ids) - covered
    print(f"  Cache: {cache_file}")
    print(f"  Covers {covered}/{len(required_doc_ids)} required docs ({missing} missing)")

    return cache_by_id, cache_file


def _run_fresh_extraction_for_missing(missing_doc_ids, mode):
    """Call GPT-4o-mini for docs not in the cache."""
    if not missing_doc_ids:
        return {}

    from dotenv import load_dotenv
    from openai import OpenAI
    from tqdm import tqdm
    from llm_extract_v2 import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, DOCUMENTS_DIR

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        print("  [warning] OPENAI_API_KEY not set - missing docs will get empty predictions")
        return {}

    client = OpenAI(api_key=api_key)

    def build_messages(abstract):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if mode == "few_shot":
            for ex in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user", "content": f"Abstract:\n{ex['abstract']}"})
                messages.append({"role": "assistant", "content": json.dumps(ex["answer"], indent=2)})
        messages.append({"role": "user", "content": f"Abstract:\n{abstract}"})
        return messages

    new_records = {}
    print(f"  Fetching {len(missing_doc_ids)} missing docs from API...")
    for doc_id in tqdm(missing_doc_ids, desc=f"LLM fresh ({mode})"):
        text_file = DOCUMENTS_DIR / f"{doc_id}.txt"
        if not text_file.exists():
            new_records[doc_id] = {"doc_id": doc_id, "abstract": "",
                                    "extracted": {"participants": [], "interventions": [], "outcomes": []}}
            continue

        with open(text_file, "r", encoding="utf-8") as f:
            abstract = f.read().strip()

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=build_messages(abstract),
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            extracted = json.loads(response.choices[0].message.content)
            for k in ["participants", "interventions", "outcomes"]:
                extracted.setdefault(k, [])
        except Exception as e:
            print(f"  [error] {doc_id}: {e}")
            extracted = {"participants": [], "interventions": [], "outcomes": []}

        new_records[doc_id] = {"doc_id": doc_id, "abstract": abstract, "extracted": extracted}

    return new_records


def _append_to_cache(new_records, mode):
    """Append new extractions to the cache for future runs."""
    if not new_records:
        return

    cache_file = _find_any_cache(mode)
    if cache_file is None:
        cache_file = LLM_OUTPUT_DIR / f"extracted_v2_{mode}_{len(new_records)}docs.json"
        existing = []
    else:
        with open(cache_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_ids = {r["doc_id"] for r in existing}
    for doc_id, rec in new_records.items():
        if doc_id not in existing_ids:
            existing.append(rec)

    LLM_OUTPUT_DIR.mkdir(exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  Cache updated: {cache_file} ({len(existing)} total docs)")


# ============================================================
# Text span -> BIO label alignment
# ============================================================

def _find_span_positions(span_text, tokens):
    span_words = span_text.lower().split()
    if not span_words:
        return []

    lower = [t.lower() for t in tokens]
    n_span = len(span_words)
    positions = []

    for i in range(len(lower) - n_span + 1):
        if lower[i : i + n_span] == span_words:
            positions.append((i, i + n_span))

    if not positions:
        joined = " ".join(lower)
        lower_span = span_text.lower()
        if lower_span in joined:
            char_start = joined.index(lower_span)
            start = len(joined[:char_start].split())
            end = start + n_span
            if end <= len(tokens):
                positions.append((start, end))

    return positions


def _spans_to_bio(span_texts, tokens):
    bio = ['O'] * len(tokens)
    for span_text in span_texts:
        for start, end in _find_span_positions(span_text, tokens):
            if bio[start] == 'O':
                bio[start] = 'B'
            for k in range(start + 1, end):
                if bio[k] == 'O':
                    bio[k] = 'I'
    return bio


# ============================================================
# Public Interface
# ============================================================

def run_separate_llm(label_type='participants', mode='few_shot', use_cached=True, allow_fresh=True):
    """
    LLM-based equivalent of run_separate().

    Args:
        label_type:  'participants' | 'interventions' | 'outcomes'
        mode:        'zero_shot' | 'few_shot'
        use_cached:  if True, try to load from llm_outputs/ first
        allow_fresh: if True, fetch missing docs from API; else pred empty for missing

    Returns:
        (test_labels, pred_labels) - list[list[str]] of BIO tags
    """
    print(f"[LLM-{mode}] Loading gold test data ({label_type})...")
    test_doc_ids = get_doc_ids(split="test", label_type=label_type)
    gold_labels, gold_tokens, _ = get_all(test_doc_ids, label_type, 'test')
    n_docs = len(test_doc_ids)
    print(f"[LLM-{mode}] {n_docs} test docs for {label_type}")

    cache_by_id = {}
    if use_cached:
        cache_by_id, _ = _load_cached_extractions_by_docids(mode, test_doc_ids)

    missing_ids = [d for d in test_doc_ids if d not in cache_by_id]
    if missing_ids and allow_fresh:
        new_records = _run_fresh_extraction_for_missing(missing_ids, mode)
        cache_by_id.update(new_records)
        _append_to_cache(new_records, mode)

    pred_labels = []
    unmatched_total = 0

    for doc_id, tokens in zip(test_doc_ids, gold_tokens):
        rec = cache_by_id.get(doc_id)
        if rec is None:
            pred_labels.append(['O'] * len(tokens))
            continue

        spans = rec["extracted"].get(label_type, [])
        bio = _spans_to_bio(spans, tokens)
        pred_labels.append(bio)

        for span_text in spans:
            if not _find_span_positions(span_text, tokens):
                unmatched_total += 1

    print(f"[LLM-{mode}] Aligned {n_docs} docs, {unmatched_total} spans could not be matched")
    return gold_labels, pred_labels