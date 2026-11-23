# ServeSmart Phase 1 corrected ETL — 2026-11-22

This is a versioned deliverable prepared from the uploaded `04_ETL_Pipeline_Final.zip` because macOS privacy controls blocked direct writes to the original Documents project. The original script and raw workbook were not overwritten.

## Run

```bash
python run_dimensional_etl_phase1.py
```

The script reads `ServeSmart Final Dataset.source_copy.xlsx` and writes reports/tables to `outputs_phase1_20261122/` when the source copy is renamed to `ServeSmart Final Dataset.xlsx`, or when `SERVESMART_SOURCE` and `SERVESMART_DIMENSIONAL_OUTPUT` are set explicitly.

## Corrections

- Preserves all register rows with `SourceSheet|row=SourceRowNumber` identity; no `drop_duplicates()` is used.
- Computes `NetAmount = Amount - Discount` and regenerates hourly summaries from the transaction fact.
- Separates `CampusClosedFlag` from outlet status. POS activity is `observed_sales`; a zero-unit/zero-net transaction hour is `observed_zero`; no outlet schedule was available, so unsupported closure is never invented and missing/unconfirmed remains unknown.
- Retains the SCD Type 4 design-gap note because the workbook has no historical effective-dated context versions.

## Verification

| Check | Result |
|---|---|
| POS rows | PASS — 65,885 |
| Item units | PASS — 63,891 |
| Gross sales | PASS — $369,654.82 |
| Discounts | PASS — $68.07 |
| Net sales | PASS — $369,586.75 |
| Source-to-fact reconciliation | PASS |
| Hourly-to-fact reconciliation | PASS |
| ETL validation | PASS — 20 checks |
| Hour-status audit | PASS — 2,030 observed_sales, 2 observed_zero, 0 unsupported closed |
| Jan 19 MLK audit | PASS — 4 location-hours, 8 units, $95.06; all observed_sales with CampusClosedFlag retained |

Remaining limitation: outlet-specific operating schedules are not present in the source workbook, so this revision cannot certify official open/closed hours. Historical SCD4 backfill also remains unsupported without dated official snapshots.
