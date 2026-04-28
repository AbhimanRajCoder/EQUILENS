import os
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")

# Path to the bundled 500-row Adult Income sample
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_SAMPLE_CSV = os.path.join(_DATA_DIR, "adult_sample.csv")


@router.get("/demo-dataset")
async def demo_dataset():
    """Return the bundled Adult Income sample as JSON + recommended column config."""
    if not os.path.exists(_SAMPLE_CSV):
        raise HTTPException(
            status_code=404,
            detail=f"Demo dataset not found at {_SAMPLE_CSV}. "
                   "Run the sample-generation script first.",
        )

    try:
        df = pd.read_csv(_SAMPLE_CSV)
        return {
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "config": {
                "target_col": "income",
                "sensitive_col": "sex",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load demo dataset: {str(e)}")
