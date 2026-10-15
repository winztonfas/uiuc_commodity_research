"""Estimate Databento requests separately; save each quote for resumable runs."""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import databento as db
import pandas as pd
from databento.common.error import BentoServerError
from dotenv import load_dotenv

from fetch.bento_fetch import (
    PROJECT_DIR, DATA_DIR, CACHE_DIR, PLAN_PATH,
    ROOTS, TABLES, build_query, request_id,
)


def save_plan(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(list(items.values()), indent=2), encoding="utf-8")
    temporary.replace(path)


def get_cost_with_retry(client, query, attempts=4):
    for attempt in range(attempts):
        try:
            return float(client.metadata.get_cost(**query))
        except BentoServerError:
            if attempt == attempts - 1:
                raise
            delay = 2 ** (attempt + 1)
            print(f"Quote server error; retrying in {delay}s...", flush=True)
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-10-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--roots", nargs="+", default=ROOTS)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--refresh", action="store_true", help="Recalculate saved quotes")
    args = parser.parse_args()
    if set(args.roots) - set(ROOTS):
        parser.error("Unknown product roots")
    if len(args.roots) != len(set(args.roots)):
        parser.error("Do not repeat roots")
    if pd.Timestamp(args.start, tz="UTC") >= pd.Timestamp(args.end, tz="UTC"):
        parser.error("--start must be before --end")

    load_dotenv(PROJECT_DIR / ".env")
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is missing from .env")
    client = db.Historical(key=key)
    items = {}
    if args.plan.exists():
        for item in json.loads(args.plan.read_text(encoding="utf-8")):
            if item["request_id"] != request_id(item["query"]):
                raise RuntimeError("Invalid request ID in existing estimate plan")
            items[item["request_id"]] = item

    total = 0.0
    for root in args.roots:
        for schema in TABLES:
            query = build_query(root, schema, args.start, args.end)
            query_id = request_id(query)
            if query_id in items and not args.refresh:
                item = items[query_id]
                label = "CACHED"
            else:
                cost = get_cost_with_retry(client, query)
                if not math.isfinite(cost) or cost < 0:
                    raise RuntimeError("Invalid cost returned by API")
                item = {
                    "root": root,
                    "schema": schema,
                    "query": query,
                    "request_id": query_id,
                    "quoted_cost_usd": cost,
                    "estimated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                items[query_id] = item
                # Save immediately so a timeout doesn't discard earlier quotes.
                save_plan(args.plan, items)
                label = "QUOTE"
            cost = float(item["quoted_cost_usd"])
            total += cost
            print(f"{label} {root:>3} {schema:<12} ${cost:.6f}", flush=True)

    print(f"\nSelected request quote total: ${total:.2f}")
    print(f"Saved estimates: {args.plan}")
    print("No market data downloaded. Saved quotes can become stale; use --refresh if needed.")


if __name__ == "__main__":
    main()
