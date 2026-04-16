# TODO

## Export pipeline (exp_04 threshold sweep follow-up)

### Switch TB exports from CSV to Parquet
- **Why:** Current CSVs are ~650 MB each (long format: 10.8M rows × 60 bytes). Parquet with same data should be ~20-40 MB (binary + columnar compression). 15-20× smaller, much faster to load.
- **Where:** `dextrah_lab/docs/export_runs.py` — single-line change in `export_tb_run()`:
  ```python
  df.to_parquet(output_csv.with_suffix('.parquet'))
  ```
- **Benefit:** Size tractable for git, plotting scripts load in seconds instead of minutes.

### Optional further reductions if still too large
1. **Drop per-object metrics** (cuts ~80% of rows if only aggregates are needed for thesis plots).
2. **Subsample time** — every 10th step instead of every step. Most plots don't need 90k-step resolution.
3. **Wide format** (pivot metric columns) — ~40% smaller than long CSV, but still larger than parquet.

### Consider for plotting scripts
- Update `plot_results.py` to read parquet if available, fall back to CSV.
- Keep CSV export as option for tools that don't handle parquet.

---

## Other pending items

_Add more as they come up._
