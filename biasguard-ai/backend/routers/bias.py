from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import asyncio
import pandas as pd
import io
from services.detector import BiasDetector
from services.explainer import BiasExplainer
from services.groq_advisor import GroqAdvisor
from models.schemas import BiasDetectionResponse

router = APIRouter(prefix="/api")
advisor = GroqAdvisor()

@router.post("/detect", response_model=BiasDetectionResponse)
async def detect_bias(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    sensitive_col: str = Form(...),
    intersectional_cols: Optional[str] = Form(None)
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

    # Parse intersectional columns (comma-separated string → list)
    i_cols = None
    if intersectional_cols:
        i_cols = [c.strip() for c in intersectional_cols.split(",") if c.strip()]
        missing = [c for c in i_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Intersectional columns not found: {', '.join(missing)}. "
                       f"Available columns: {', '.join(df.columns.tolist())}"
            )

    # Initialize and run detector
    try:
        detector = BiasDetector(df, target_col, sensitive_col)
        results = detector.run_detection()
        
        # Initialize and run explainer
        explainer = BiasExplainer(
            model=results["model"],
            X_train=results["X_train"],
            feature_names=results["feature_names"],
            df=df,
            sensitive_col=sensitive_col
        )
        explanation_results = explainer.get_explanations()
        
        # Combine results
        combined_results = {
            "accuracy": results["accuracy"],
            "per_group_approval_rates": results["per_group_approval_rates"],
            "fairness_metrics": results["fairness_metrics"],
            "shap_features": explanation_results["shap_features"],
            "per_group_shap": explanation_results.get("per_group_shap"),
            "group_comparison": explanation_results["group_comparison"],
            "advanced_metrics": results.get("advanced_metrics")
        }

        # Intersectional analysis (optional)
        if i_cols and len(i_cols) >= 1:
            combined_results["intersectional_bias"] = (
                detector.run_intersectional_analysis(i_cols)
            )

        # Counterfactual explanations (always attempted)
        top_shap_names = [f["feature"] for f in explanation_results["shap_features"][:5]]
        combined_results["counterfactual_examples"] = (
            detector.run_counterfactual_analysis(top_shap_names)
        )

        # Groq narratives (additive only; never block audit flow)
        combined_results["groq_narrative"] = None
        combined_results["groq_shap_insight"] = None
        if advisor.is_configured():
            try:
                narrative_task = advisor.generate_bias_narrative(
                    metrics=results["fairness_metrics"],
                    sensitive_col=sensitive_col,
                    dataset_name=file.filename or "uploaded dataset",
                )
                shap_task = advisor.explain_shap_features(
                    shap_data=explanation_results["shap_features"],
                    sensitive_col=sensitive_col,
                )
                groq_narrative, groq_shap_insight = await asyncio.gather(
                    narrative_task, shap_task, return_exceptions=True
                )
                if not isinstance(groq_narrative, Exception):
                    combined_results["groq_narrative"] = groq_narrative
                if not isinstance(groq_shap_insight, Exception):
                    combined_results["groq_shap_insight"] = groq_shap_insight
            except Exception:
                # Graceful degradation: keep base detection response intact.
                pass
        
        return combined_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
