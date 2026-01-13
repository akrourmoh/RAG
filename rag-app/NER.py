from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline
from dateparser.search import search_dates


def run_ner(text: str) -> List[Dict[str, Any]]:
    """
    Runs Named Entity Recognition using a pretrained Transformer model.
    Note: This specific model is typically trained on PER/ORG/LOC/MISC (not DATE).
    """
    ner_pipeline = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )
    return ner_pipeline(text)


def extract_dates(text: str, languages: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Extracts date mentions from text using dateparser.search.search_dates.
    Returns a list with the matched text and parsed datetime.
    """
    languages = languages or ["en"]

    matches: Optional[List[Tuple[str, Any]]] = search_dates(
        text,
        languages=languages,
        settings={
            # Prefer interpreting ambiguous dates as DMY/MDY depending on your needs:
            # "DATE_ORDER": "DMY",
            "PREFER_DAY_OF_MONTH": "first",
        },
    )

    results: List[Dict[str, Any]] = []
    if not matches:
        return results

    for matched_text, dt in matches:
        results.append({
            "matched_text": matched_text,
            "parsed_datetime": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
        })

    return results


def main() -> None:
    # ✅ Put your own text here
    text = "John Smith was admitted to Lille University Hospital on March 12, 2024."

    print("TEXT:")
    print(text)
    print("\n--- NER (Transformer) ---")
    entities = run_ner(text)
    for ent in entities:
        # ent includes: entity_group, score, word, start, end
        print(ent)

    print("\n--- Dates (dateparser) ---")
    dates = extract_dates(text, languages=["en"])
    if not dates:
        print("No dates detected.")
    else:
        for d in dates:
            print(d)

    print("\n--- Combined summary ---")
    summary = {
        "entities": entities,
        "dates": dates,
    }
    print(summary)


if __name__ == "__main__":
    main()
