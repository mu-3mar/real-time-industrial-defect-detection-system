# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import sqlite3
from pathlib import Path
from datetime import datetime
import csv
import io

# ================== CONFIG ==================
DB_PATH = "../flow/outputs/products.db"            # The same database used by the pipeline
SNAPSHOT_DIR = "../flow/outputs/defect_snapshots"  # The same snapshot folder (use if desired)
# ============================================

app = FastAPI(title="QC Pipeline API", version="1.1")


# ----------------- DB helper -----------------
def get_db_conn():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ----------------- HEALTH -----------------
@app.get("/health")
async def health_check():
    info = {
        "db_exists": Path(DB_PATH).exists(),
        "snapshot_dir_exists": Path(SNAPSHOT_DIR).exists(),
    }
    return JSONResponse(content=info, status_code=200)


# =========================================================
#      NEW: TIME-BASED SUMMARY (no session_id)
# =========================================================
@app.get("/summary")
async def get_summary(start: str | None = None, end: str | None = None):
    """
    Time-based summary endpoint.

    Behavior:
      - if only `start` is provided: from `start` to the latest timestamp in the DB
      - if only `end` is provided: from the earliest timestamp in the DB to `end`
      - if both provided: between `start` and `end`
      - if neither provided: include all data

    Query params (ISO format recommended):
      /summary?start=2025-11-23T00:00:00&end=2025-11-23T23:59:59
      /summary?start=2025-11-23
      /summary?end=2025-11-23T12:00:00
    """

    # 1) Open the database
    try:
        conn = get_db_conn()
    except FileNotFoundError:
        return JSONResponse(content={"error": "Database not found"}, status_code=404)

    cur = conn.cursor()

    # 2) Retrieve the minimum and maximum timestamps present in the DB
    cur.execute("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM products")
    row = cur.fetchone()
    if not row or row["min_ts"] is None:
        conn.close()
        return JSONResponse(content={"error": "No data in database"}, status_code=404)

    db_min_ts = row["min_ts"]
    db_max_ts = row["max_ts"]

    # 3) Parse `start`/`end` if they were provided
    def parse_ts(ts_str: str) -> str:
        """
        Perform a simple validation using `datetime.fromisoformat` and
        return the string as-is (SQLite compares ISO datetime strings correctly).
        """
        try:
            datetime.fromisoformat(ts_str)
            return ts_str
        except Exception:
            # The user might send a date-only string such as "2025-11-23".
            # `fromisoformat` handles that; if parsing fails, raise an error.
            raise ValueError(f"Invalid datetime format: {ts_str}")

    try:
        if start is not None:
            start_ts = parse_ts(start)
        else:
            start_ts = db_min_ts

        if end is not None:
            end_ts = parse_ts(end)
        else:
            end_ts = db_max_ts
    except ValueError as e:
        conn.close()
        return JSONResponse(content={"error": str(e)}, status_code=400)

    # Ensure start <= end
    if start_ts > end_ts:
        conn.close()
        return JSONResponse(content={"error": "start must be <= end"}, status_code=400)

    # 4) Fetch products within this time range
    cur.execute(
        """
        SELECT product_id, final_status, max_defects, first_frame, last_frame, frames_seen, timestamp, snapshot_path
        FROM products
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """,
        (start_ts, end_ts),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return JSONResponse(
            content={
                "start_applied": start_ts,
                "end_applied": end_ts,
                "total_products": 0,
                "defective_count": 0,
                "non_defective_count": 0,
                "defect_rate": 0.0,
                "products": [],
            },
            status_code=200,
        )

    # 5) Compute summary statistics
    total_products = len(rows)
    defective_count = sum(1 for r in rows if r["final_status"] == "defect")
    non_defective_count = total_products - defective_count
    defect_rate = (defective_count / total_products) * 100.0 if total_products > 0 else 0.0

    # If desired, return details for each product
    products = []
    for r in rows:
        products.append(
            {
                "product_id": r["product_id"],
                "final_status": r["final_status"],
                "max_defects": r["max_defects"],
                "first_frame": r["first_frame"],
                "last_frame": r["last_frame"],
                "frames_seen": r["frames_seen"],
                "timestamp": r["timestamp"],
                "snapshot_path": r["snapshot_path"],
            }
        )

    return JSONResponse(
        content={
            "start_applied": start_ts,      # actual applied range
            "end_applied": end_ts,
            "total_products": total_products,
            "defective_count": defective_count,
            "non_defective_count": non_defective_count,
            "defect_rate": defect_rate,     # defect rate (%)
            "products": products,           # if large, consider removing this list
        },
        status_code=200,
    )


# =========================================================
#    Legacy endpoints (optional — keep or remove as needed)
# =========================================================

@app.get("/download-csv-by-time")
async def download_csv_by_time(start: str | None = None, end: str | None = None):
    """
    Similar behavior to `/summary` but returns results as CSV instead of JSON.
    """
    # Same `start`/`end` logic as in `/summary`
    try:
        conn = get_db_conn()
    except FileNotFoundError:
        return JSONResponse(content={"error": "Database not found"}, status_code=404)

    cur = conn.cursor()
    cur.execute("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM products")
    row = cur.fetchone()
    if not row or row["min_ts"] is None:
        conn.close()
        return JSONResponse(content={"error": "No data in database"}, status_code=404)

    db_min_ts = row["min_ts"]
    db_max_ts = row["max_ts"]

    def parse_ts(ts_str: str) -> str:
        from datetime import datetime
        try:
            datetime.fromisoformat(ts_str)
            return ts_str
        except Exception:
            raise ValueError(f"Invalid datetime format: {ts_str}")

    try:
        if start is not None:
            start_ts = parse_ts(start)
        else:
            start_ts = db_min_ts

        if end is not None:
            end_ts = parse_ts(end)
        else:
            end_ts = db_max_ts
    except ValueError as e:
        conn.close()
        return JSONResponse(content={"error": str(e)}, status_code=400)

    if start_ts > end_ts:
        conn.close()
        return JSONResponse(content={"error": "start must be <= end"}, status_code=400)

    cur.execute(
        """
        SELECT product_id, session_id, final_status, max_defects,
               first_frame, last_frame, frames_seen, timestamp, snapshot_path
        FROM products
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """,
        (start_ts, end_ts),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return JSONResponse(content={"error": "No products in this time range"}, status_code=404)

    def iter_csv():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "product_id",
                "session_id",
                "final_status",
                "max_defects",
                "first_frame",
                "last_frame",
                "frames_seen",
                "timestamp",
                "snapshot_path",
            ]
        )
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        for r in rows:
            writer.writerow(
                [
                    r["product_id"],
                    r["session_id"],
                    r["final_status"],
                    r["max_defects"],
                    r["first_frame"],
                    r["last_frame"],
                    r["frames_seen"],
                    r["timestamp"],
                    r["snapshot_path"],
                ]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    filename = f"products_{start_ts}_to_{end_ts}.csv".replace(":", "-")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(iter_csv(), media_type="text/csv", headers=headers)


# You may keep `/product` and `/sessions` endpoints if needed, or remove them
# if you don't want session_id-based APIs.


# ================== RUN INSTRUCTIONS ==================
# Run the server:
#   pip install fastapi uvicorn
#   uvicorn api_server:app --host 0.0.0.0 --port 8000
#
# Usage examples:
#   /summary                    -> return all data
#   /summary?start=2025-11-23   -> from this date to the latest data
#   /summary?end=2025-11-23T12:00:00
#   /summary?start=2025-11-23T00:00:00&end=2025-11-23T23:59:59
#
#   /download-csv-by-time?start=...&end=...
