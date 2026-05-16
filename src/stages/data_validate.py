"""
Stage 1.5: Data Validation
Uses Pandera to enforce schema and data quality rules on the raw IBM Telco dataset.
Raises SchemaErrors if constraints are violated — halting the DVC pipeline early.
"""

import sys
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import yaml
from loguru import logger
from pathlib import Path


# ── Define Schema ─────────────────────────────────────────────────────────────
schema = DataFrameSchema(
    {
        "customerID":      Column(str, unique=True, nullable=False),
        "gender":          Column(str, Check.isin(["Female", "Male"])),
        "SeniorCitizen":   Column(int, Check.isin([0, 1])),
        "tenure":          Column(int, Check.ge(0)),        # must be non-negative
        "MonthlyCharges":  Column(float, Check.ge(0.0)),
        "TotalCharges":    Column(str),                      # raw string — spaces for new customers
        "Churn":           Column(str, Check.isin(["Yes", "No"])),
        "Contract":        Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "PaymentMethod":   Column(
            str,
            Check.isin([
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ]),
        ),
    },
    coerce=False,
    strict=False,   # allow extra columns not listed above
)


def main() -> None:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_path = params["data"]["raw_data_path"]
    logger.info(f"Validating raw data at: {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    try:
        schema.validate(df, lazy=True)
        logger.success(
            "Data validation PASSED ✓ — all schema constraints satisfied."
        )

        Path("reports").mkdir(exist_ok=True)
        with open("reports/data_validation_status.txt", "w") as f:
            f.write(f"PASSED\nRows: {len(df)}\nColumns: {len(df.columns)}\n")

    except pa.errors.SchemaErrors as err:
        logger.error("Data validation FAILED ✗")
        logger.error(f"\n{err.failure_cases.to_string()}")

        Path("reports").mkdir(exist_ok=True)
        with open("reports/data_validation_status.txt", "w") as f:
            f.write("FAILED\n")
            f.write(err.failure_cases.to_string())

        # Fail loudly so DVC stops the pipeline
        sys.exit(1)


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/data_validate.log", rotation="1 MB")
    main()
