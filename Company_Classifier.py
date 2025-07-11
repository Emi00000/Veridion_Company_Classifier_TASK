from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

DEFAULT_FEWSHOT_BLOCK = """
Example 1
Taxonomy:

["Branding Services", "Interior Design Services", "Office Furniture Design and Manufacturing"]
Company:

Description: "SellerGroup specialises in promotional and office products. They design and manufacture customised desks, chairs and conference tables, plus ergonomic solutions (ErgoDesk, CleanDesk) and workplace-branding advice."
Business tags: ["Office Furniture Design and Manufacturing", "Workplace Solutions", "Hygiene Products", "Promotional Products with Logos", "Desk Cleaning", "Branding Services", "Branding Enhancement", "Conference Tables"]
Sector: "Manufacturing"
Category: "Office Furniture"
Niche: "Wood Office Furniture Manufacturing"
Assistant:

["Branding Services"]

Example 2
Taxonomy:

["Interior Design Services", "Urban Design Services", "Architectural Design and Implementation Services"]
Company:

Description: "Design firm focused on improving quality of life via urban planning, architecture, landscape and interior design."
Business tags: ["Interior Design Services", "Urban Design Services", "Architectural Design and Implementation Services", "Landscape Design Services"]
Sector: "Services"
Category: "Architects & Architectural Services"
Niche: "Architectural Services"
Assistant:

["Interior Design Services"]

Example 3
Taxonomy:

["Carpentry Services", "Motor Vehicle Body Manufacturing", "Trailer Manufacturer"]
Company:

Description: "Fahrzeugbau-pfaff.de offers specialised vehicle construction and motor-home builds, including trailers and special commercial vehicles."
Business tags: ["Vehicle Construction Services", "Construction Services", "Motorhome Construction", "Commercial Vehicles Manufacturer", "Trailer Manufacturer", "Boat Trailer Design and Construction", "Transfer Trucks", "Caravans Manufacturer", "Carpentry Services", "Commercial Vehicles for Construction", "Mobile Construction Services", "Media Mobile Vehicles Manufacturer", "Special Vehicles Manufacturer"]
Sector: "Manufacturing"
Category: "Auto Parts Manufacturers"
Niche: "Motor Vehicle Body Manufacturing"
Assistant:

["Carpentry Services"]

Example 4
Taxonomy:

["Confectionery Manufacturing", "Baking Ingredients", "Private Label Manufacturing"]
Company:

Description: "Mago Indústria, Brazil’s largest confectionery supplier, produces colourants, sprinkles, baking tools and accessories while offering courses and recipes."
Business tags: ["Conveying Products", "Baking Ingredients", "Cosmetics Manufacturer", "Candy Bags", "Conveyor Confections", "Cake in The Pot", "Food Delivery", "Private Label Manufacturing", "Credit Card Payment", "Colorant Products", "Bathroom Facilities", "Educational Courses", "Decoration Powdering", "Tools and Accessories", "Sustainable Manufacturing", "Confectionery Sprinkles", "Confectionery Manufacturing", "Brand Development Services", "Personal Care Products Manufacturer", "Pastry Shop", "Wheelchair Accessible Entrance"]
Sector: "Manufacturing"
Category: "Bakeries & Desserts"
Niche: "Nonchocolate Confectionery Manufacturing"
Assistant:

["Confectionery Manufacturing"]
"""


LOGGER = logging.getLogger("taxonomy")
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

class RequireOpenBracket(LogitsProcessor):
    def __init__(self, prompt_len: int, open_id: int) -> None:
        self.prompt_len, self.open_id = prompt_len, open_id

    def __call__(self, ids, scores):
        if ids.shape[1] == self.prompt_len:  # first generation step
            scores[:] = -float("inf")
            scores[0, self.open_id] = 0.0
        return scores



def load_embeddings(
    taxonomy_csv: Path, embedder: SentenceTransformer
) -> tuple[list[str], torch.Tensor]:
    labels = (
        pd.read_csv(taxonomy_csv)
        .iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )
    LOGGER.info("Encoding %d taxonomy labels …", len(labels))
    emb = embedder.encode(labels, convert_to_tensor=True, normalize_embeddings=True)
    return labels, emb


def parse_labels(raw: str, valid: set[str], max_out: int = 3) -> List[str]:
    match = re.search(r"\[.*?\]", raw, flags=re.S)
    payload = match.group(0) if match else raw

    try:
        obj = json.loads(payload)
    except Exception:
        obj = re.split(r"[,\n]", payload.strip("[]{}() "))

    if isinstance(obj, str):
        items: Sequence[str] = [obj]
    elif isinstance(obj, dict):
        items = obj.get("labels") or obj.get("label") or obj.values()
    else:
        items = obj

    cleaned: list[str] = []
    for it in items:
        it = str(it).strip(" \"'").lower()
        if it and it in valid:
            cleaned.append(it)

    return list(dict.fromkeys(cleaned))[:max_out]


def build_prompt(row: pd.Series, cand: list[str], fewshot_block: str) -> str:
    lbls = ", ".join(f'"{l}"' for l in cand)
    return f"""<|begin_of_text|>
### System:
You are an insurance-taxonomy assistant.
Return **ONLY** a JSON list (1–3 items) of taxonomy labels, chosen from **Taxonomy**, that fit the company.

{fewshot_block}

### Task
### Taxonomy:
[{lbls}]

### Company:
Description: "{row['description']}"
Business tags: {row['business_tags']}
Sector: "{row['sector']}"
Category: "{row['category']}"
Niche: "{row['niche']}"

<|assistant|>"""

@torch.inference_mode()
def main(
    companies_csv: Path,
    taxonomy_csv: Path,
    outfile: Path,
    rows: int,
    batch: int,
    device: str,
    fewshot_block: str,
) -> None:
    t0 = time.time()
    MODEL_ID = "unsloth/llama-3-8b-Instruct-bnb-4bit"

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=device, torch_dtype=torch.bfloat16
    )
    open_id = tok.encode("[", add_special_tokens=False)[0]
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")

    label_list, label_embs = load_embeddings(taxonomy_csv, embedder)
    label_set = {l.lower() for l in label_list}
    top_k = 60

    df = pd.read_csv(companies_csv).fillna("").head(rows)
    LOGGER.info("Loaded %d company rows", len(df))

    texts = (
        df["description"].astype(str)
        + " "
        + df["business_tags"].astype(str)
        + " "
        + df["sector"].astype(str)
        + " "
        + df["category"].astype(str)
        + " "
        + df["niche"].astype(str)
    ).tolist()

    company_embs = embedder.encode(
        texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch
    )
    LOGGER.info("Embeddings ready (%.1fs)", time.time() - t0)

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predict"):
        q_emb = company_embs[idx]
        cand_idx = torch.topk(q_emb @ label_embs.T, top_k).indices.tolist()
        cand = [label_list[i] for i in cand_idx]

        prompt = build_prompt(row, cand, fewshot_block)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        lp = LogitsProcessorList([RequireOpenBracket(inp.input_ids.size(1), open_id)])

        gen = model.generate(
            **inp,
            num_beams=3,
            num_return_sequences=3,
            max_new_tokens=64,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eot_id,
            logits_processor=lp,
            do_sample=False,
        )

        parses = [
            parse_labels(
                tok.decode(seq[inp.input_ids.size(1) :], skip_special_tokens=True),
                label_set,
            )
            for seq in gen
        ]
        flat = sum(parses, [])
        maj = [lab for lab, c in Counter(flat).most_common(3) if c >= 2]
        chosen = maj or parses[0] or [cand[0].lower()]

        results.append({"predicted_labels": "; ".join(chosen), **row.to_dict()})

        del inp, gen
        torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(outfile, index=False)
    LOGGER.info("Saved predictions → %s (%.1f s)", outfile, time.time() - t0)

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict taxonomy labels for companies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--companies", type=Path, required=True, help="CSV with companies")
    p.add_argument("--taxonomy", type=Path, required=True, help="CSV with taxonomy")
    p.add_argument("--outfile", type=Path, required=True, help="Destination CSV")
    p.add_argument("--rows", type=int, default=5_000, help="Max rows to process")
    p.add_argument("--batch", type=int, default=32, help="Embedding batch size")
    p.add_argument("--device", type=str, default="auto", help="Device map for HF")
    p.add_argument(
        "--fewshot",
        type=Path,
        help=(
            "Optional text file containing few-shot examples. "
            "If omitted, a built-in template is used."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()

    if args.fewshot:
        LOGGER.info("Loading few-shot examples from %s", args.fewshot)
        fewshot_text = args.fewshot.read_text(encoding="utf-8")
    else:
        fewshot_text = DEFAULT_FEWSHOT_BLOCK

    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    try:
        main(
            companies_csv=args.companies,
            taxonomy_csv=args.taxonomy,
            outfile=args.outfile,
            rows=args.rows,
            batch=args.batch,
            device=args.device,
            fewshot_block=fewshot_text,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user, exiting…")
