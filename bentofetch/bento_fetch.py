"""
This it's mainly an AI generated fetching version, since the time for the ml
project kinda hurrying I will clean the system up later..
"""


import argparse
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import databento as db
from databento.common.error import BentoError, BentoServerError, BentoClientError
import duckdb
import pandas as pd
from dotenv import load_dotenv

# Create a historical client

start = "2021-10-01"
end = "2026-10-01"

## examinate the cost 
# print("Current schema prices:")
# print(client.metadata.list_unit_prices(dataset="GLBX.MDP3"))

# total = 0.0

# for root in us_roots:
#     for schema in ("ohlcv-1d", "statistics", "definition"):
#         try:
#             cost = float(client.metadata.get_cost(
#                 dataset="GLBX.MDP3",
#                 symbols=[f"{root}.FUT"],
#                 stype_in="parent",
#                 schema=schema,
#                 start=start,
#                 end=end,
#             ))
#         except Exception as exc:
#             print(f"{root:>4} {schema:<12} QUOTE FAILED: {exc}")
#             continue

#         total += cost
#         print(f"{root:>4} {schema:<12} ${cost:.4f}")

# print(f"\nTotal for successful quotes: ${total:.2f}")
# print("Failed requests are excluded; this is not a complete quote if any failed.")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "databento"
CACHE_DIR = DATA_DIR / "dbn"
DATABASE_PATH = DATA_DIR / "commodities.duckdb"
PLAN_PATH = DATA_DIR / "download_plan.json"

DATASET = "GLBX.MDP3"

ROOTS = [
    # Metals
    "GC", "SI", "HG", "ALI", "HRC",
    # Energy
    "CL", "HO", "RB", "NG", "PL", "PA",
    # Agriculture
    "ZS", "ZM", "ZL", "ZC", "ZW", "KE", "ZR", "HE",
]

TABLES = {
    "ohlcv-1d": "raw_ohlcv_1d",
}


def quoted_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def build_query(root, schema, start, end):
    return {
        "dataset": DATASET,
        "symbols": [f"{root}.FUT"],
        "stype_in": "parent",
        "schema": schema,
        "start": start,
        "end": end,
    }


def request_id(query):
    payload = json.dumps(query, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def initialize_database(connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS download_manifest (
            request_id VARCHAR PRIMARY KEY,
            dataset VARCHAR,
            root VARCHAR,
            schema_name VARCHAR,
            start_utc VARCHAR,
            end_utc VARCHAR,
            quoted_cost_usd DOUBLE,
            row_count BIGINT,
            dbn_path VARCHAR,
            completed_at TIMESTAMPTZ
        )
    """)


def is_complete(connection, query_id):
    return connection.execute(
        """
        SELECT COUNT(*)
        FROM download_manifest
        WHERE request_id = ?
        """,
        [query_id],
    ).fetchone()[0] > 0


def normalize_frame(frame, root, query_id):
    frame = frame.reset_index()

    # pretty_ts=False preserves original integer nanosecond values.
    timestamp_columns = [
        name for name in frame.columns
        if name.startswith("ts_")
        or name in {"expiration", "activation"}
    ]
    frame = frame.rename(columns={
        name: f"{name}_ns" for name in timestamp_columns
    })

    # Avoid DuckDB inferring an integer type for all-null text fields.
    for name in frame.columns:
        if (
            pd.api.types.is_object_dtype(frame[name].dtype)
            or isinstance(frame[name].dtype, pd.CategoricalDtype)
        ):
            frame[name] = frame[name].astype("string")

    frame["source_dataset"] = DATASET
    frame["product_root"] = root
    frame["source_request_id"] = query_id
    return frame


def append_frame(connection, table, frame):
    connection.register("_incoming", frame)

    try:
        table_sql = quoted_identifier(table)

        connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_sql}
            AS SELECT * FROM _incoming WHERE FALSE
        """)

        existing_columns = {
            row[1]
            for row in connection.execute(
                f"PRAGMA table_info('{table}')"
            ).fetchall()
        }

        # Allow additional fields from newer definition record versions.
        incoming_columns = connection.execute(
            "DESCRIBE SELECT * FROM _incoming"
        ).fetchall()

        for name, dtype, *_ in incoming_columns:
            if name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE {table_sql} "
                    f"ADD COLUMN {quoted_identifier(name)} {dtype}"
                )

        connection.execute(f"""
            INSERT INTO {table_sql} BY NAME
            SELECT * FROM _incoming
        """)
    finally:
        connection.unregister("_incoming")


def download_with_retry(client, query, partial_path, attempts=4):
    for attempt in range(1, attempts + 1):
        if partial_path.exists():
            partial_path.unlink()
        try:
            return client.timeseries.get_range(**query, path=partial_path)
        except BentoError as exc:
            transient = (
                isinstance(exc, BentoServerError)
                or (type(exc) is BentoError and "Error streaming response" in str(exc))
            )
            if not transient or attempt == attempts:
                raise
            size = partial_path.stat().st_size if partial_path.exists() else 0
            delay = min(5 * 2 ** (attempt - 1), 30)
            print(
                f"  Temporary server/stream error; retry {attempt}/{attempts - 1} "
                f"in {delay}s. This attempt saved {size / 1_048_576:.2f} MiB. "
                "Retry restarts the full request; delivered bytes may be charged again.",
                flush=True,
            )
            time.sleep(delay)


def fetch_single_request(connection, client, item):
    root = item["root"]
    schema = item["schema"]
    query = item["query"]
    query_id = item["request_id"]
    cost = item["quoted_cost_usd"]

    cache_path = CACHE_DIR / (
        f"{root}_{schema}_{query_id[:16]}.dbn.zst"
    )
    partial_path = cache_path.with_suffix(
        cache_path.suffix + ".partial"
    )

    if cache_path.exists():
        print(f"  Using local cache: {cache_path.name}")
    else:
        # A partial download isn't safe to reuse.
        if partial_path.exists():
            partial_path.unlink()

        print(f" Downloading {root} / {schema} ...")

        finished = threading.Event()
        started = time.monotonic()

        def show_progress():
            previous = -1
            while not finished.wait(10):
                size = partial_path.stat().st_size if partial_path.exists() else 0
                state = "receiving" if size > previous else "waiting for data"
                print(
                    f"  {root}/{schema}: {size / 1_048_576:.2f} MiB, "
                    f"{time.monotonic() - started:.0f}s elapsed ({state})",
                    flush=True,
                )
                previous = size

        monitor = threading.Thread(target=show_progress, daemon=True)
        monitor.start()
        try:
            download_with_retry(client, query, partial_path)
        except BentoClientError as exc:
            if "account_insufficient_funds" in str(exc):
                raise RuntimeError(
                    "Databento rejected this request with 402 account_insufficient_funds. "
                    "Check Billing: pay-as-you-go access, remaining historical credits, "
                    "and your historical monthly usage limit. The script's --max-cost "
                    "does not change your Databento budget. No automatic retry was made. "
                    "Existing database rows have been retained."
                ) from exc
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Download interrupted: {root}/{schema} "
                f"[{query['start']}, {query['end']}). "
                "Completed products are safe. Rerun to refetch this full request; "
                "previously delivered bytes may be billed again."
            ) from exc
        finally:
            finished.set()
            monitor.join()


        # Only mark the cache complete after the API call succeeds.
        partial_path.replace(cache_path)

    store = db.DBNStore.from_file(cache_path)
    if item.get("reset_after_download"):
        cleanup_incomplete_request(connection, item, keep_cache=cache_path)
    total_rows = 0

    # Raw records and completion marker are committed together.
    connection.execute("BEGIN TRANSACTION")

    try:
        for frame in store.to_df(
            price_type="fixed",
            pretty_ts=False,
            map_symbols=True,
            count=50_000,
        ):
            if frame.empty:
                continue

            frame = normalize_frame(frame, root, query_id)
            append_frame(connection, TABLES[schema], frame)
            total_rows += len(frame)
            print(f"  Importing {root}/{schema}: {total_rows:,} rows", flush=True)

        connection.execute(
            """
            INSERT INTO download_manifest VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                query_id,
                DATASET,
                root,
                schema,
                query["start"],
                query["end"],
                cost,
                total_rows,
                str(cache_path.relative_to(PROJECT_DIR)),
                datetime.now(timezone.utc),
            ],
        )

        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise

    print(f"  Saved {total_rows:,} rows to {TABLES[schema]}")



def monthly_queries(query):
    current = pd.Timestamp(query["start"], tz="UTC")
    end_time = pd.Timestamp(query["end"], tz="UTC")
    while current < end_time:
        next_time = min(current + pd.offsets.MonthBegin(1), end_time)
        yield {**query, "start": current.isoformat(), "end": next_time.isoformat()}
        current = next_time


def cleanup_incomplete_request(connection, item, keep_cache=None):
    query = item["query"]
    root, schema = item["root"], item["schema"]
    first = pd.Timestamp(query["start"], tz="UTC").isoformat()
    last = pd.Timestamp(query["end"], tz="UTC").isoformat()
    saved = connection.execute(
        """SELECT request_id FROM download_manifest
           WHERE dataset = ? AND root = ? AND schema_name = ?
             AND CAST(start_utc AS TIMESTAMPTZ) >= CAST(? AS TIMESTAMPTZ)
             AND CAST(end_utc AS TIMESTAMPTZ) <= CAST(? AS TIMESTAMPTZ)""",
        [DATASET, root, schema, first, last],
    ).fetchall()
    ids = {row[0] for row in saved}
    ids.add(item["request_id"])
    ids.update(request_id(q) for q in monthly_queries(query))
    table = TABLES[schema]
    table_exists = connection.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()[0] > 0
    connection.execute("BEGIN TRANSACTION")
    try:
        for query_id in ids:
            if table_exists:
                connection.execute(
                    f"DELETE FROM {quoted_identifier(table)} WHERE source_request_id = ?",
                    [query_id],
                )
            connection.execute("DELETE FROM download_manifest WHERE request_id = ?", [query_id])
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    removed = 0
    for query_id in ids:
        cache = CACHE_DIR / f"{root}_{schema}_{query_id[:16]}.dbn.zst"
        for candidate in (cache, cache.with_suffix(cache.suffix + ".partial")):
            if candidate.resolve().parent != CACHE_DIR.resolve():
                raise RuntimeError("Cleanup path is outside the DBN cache directory")
            if keep_cache is not None and candidate.resolve() == keep_cache.resolve():
                continue
            if candidate.exists():
                candidate.unlink()
                removed += 1
    print(f"  Reset {root}/{schema}: removed {len(saved)} saved chunk entries and {removed} files.", flush=True)


def fetch_and_import(connection, client, item):
    if is_complete(connection, item["request_id"]):
        print("  SKIP completed full request", flush=True)
        return
    fetch_single_request(connection, client, {**item, "reset_after_download": True})


        # Monthly costs are allocations of the saved full-window quote, not new API quotes.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-10-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--roots", nargs="+", default=ROOTS)
    parser.add_argument("--max-cost", type=float, default=120.0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    args = parser.parse_args()

    unknown = set(args.roots) - set(ROOTS)
    if unknown:
        parser.error(f"Unknown roots: {sorted(unknown)}")

    # Reject duplicate roots and overlapping request windows.
    # The baseline uses one fixed window per root/schema.
    if len(args.roots) != len(set(args.roots)):
        parser.error("Do not repeat roots.")

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    if start >= end:
        parser.error("--start must be before --end")

    if args.max_cost <= 0:
        parser.error("--max-cost must be positive")

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.environ.get("DATABENTO_API_KEY")

    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY is missing from .env")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    client = db.Historical(key=api_key)
    connection = duckdb.connect(str(DATABASE_PATH))
    initialize_database(connection)

    saved_items = {}
    if args.plan.exists():
        for item in json.loads(args.plan.read_text(encoding="utf-8")):
            if item["request_id"] != request_id(item["query"]):
                raise RuntimeError("Invalid request ID in estimate plan.")
            saved_items[item["request_id"]] = item

    plan = []

    try:
        for root in args.roots:
            for schema in TABLES:
                query = build_query(
                    root, schema, args.start, args.end
                )
                query_id = request_id(query)

                if is_complete(connection, query_id):
                    print(f"SKIP {root:>3} {schema:<12} already imported")
                    continue

                # Prevent duplicate imports from overlapping date requests.
                overlap = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM download_manifest
                    WHERE dataset = ?
                      AND root = ?
                      AND schema_name = ?
                      AND CAST(start_utc AS TIMESTAMPTZ)
                          < CAST(? AS TIMESTAMPTZ)
                      AND CAST(end_utc AS TIMESTAMPTZ)
                          > CAST(? AS TIMESTAMPTZ)
                      AND NOT (
                          schema_name IN ('ohlcv-1d', 'statistics', 'definition')
                          AND CAST(start_utc AS TIMESTAMPTZ) >= CAST(? AS TIMESTAMPTZ)
                          AND CAST(end_utc AS TIMESTAMPTZ) <= CAST(? AS TIMESTAMPTZ)
                      )
                    """,
                    [
                        DATASET, root, schema,
                        end.isoformat(), start.isoformat(), start.isoformat(), end.isoformat(),
                    ],
                ).fetchone()[0]

                if overlap:
                    raise RuntimeError(
                        f"{root}/{schema}: overlapping dates already "
                        "imported. Use a non-overlapping extension window."
                    )

                cache_path = CACHE_DIR / (
                    f"{root}_{schema}_{query_id[:16]}.dbn.zst"
                )

                if query_id in saved_items:
                    saved = saved_items[query_id]
                    if saved["root"] != root or saved["schema"] != schema:
                        raise RuntimeError("Estimate plan metadata mismatch.")
                    cost = float(saved["quoted_cost_usd"])
                    if not (0 <= cost < float("inf")):
                        raise RuntimeError("Invalid cost in estimate plan.")
                else:
                    raise RuntimeError(
                        f"Missing saved estimate for {root}/{schema}. "
                        "Run bento_estimate.py with the same dates and roots first."
                    )

                plan.append({
                    "root": root,
                    "schema": schema,
                    "query": query,
                    "request_id": query_id,
                    "quoted_cost_usd": cost,
                })

                print(f"SAVED {root:>3} {schema:<12} ${cost:.6f}")

        total = sum(item["quoted_cost_usd"] for item in plan)

        print(f"\nEstimated NEW download cost: ${total:.2f}")
        print(f"Database: {DATABASE_PATH}")

        if total > args.max_cost:
            raise RuntimeError(
                f"Quote exceeds your ${args.max_cost:.2f} limit. "
                "No market data downloaded."
            )

        if not args.download:
            print("\nSaved-plan preview only. Add --download to fetch; no quotes requested.")
            return

        for index, item in enumerate(plan, start=1):
            print(
                f"\n[{index}/{len(plan)}] "
                f"{item['root']} / {item['schema']}"
            )
            fetch_and_import(connection, client, item)

        print("\nCompleted successfully.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
