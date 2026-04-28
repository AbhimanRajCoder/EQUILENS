from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd
import io
from services.simulator import BiasSimulator
from models.schemas import SimulationResponse

router = APIRouter(prefix="/api")

@router.post("/simulate", response_model=SimulationResponse)
async def simulate_bias(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    sensitive_col: str = Form(...)
):
    # Read CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        # Clean column names: strip whitespace and quotes
        df.columns = [c.strip().replace('"', '').replace("'", "") for c in df.columns]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    # Check columns
    if target_col not in df.columns or sensitive_col not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise HTTPException(
            status_code=400, 
            detail=f"Columns '{target_col}' or '{sensitive_col}' not found. Available columns: {available_cols}"
        )

    # Initialize and run simulator
    try:
        simulator = BiasSimulator(df, target_col, sensitive_col)
        results = simulator.run_simulation()
        return {"strategies": results}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
