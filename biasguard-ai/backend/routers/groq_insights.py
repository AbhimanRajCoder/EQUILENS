from fastapi import APIRouter, Body

from services.groq_advisor import GroqAdvisor

router = APIRouter(prefix="/api/groq")
advisor = GroqAdvisor()


def _missing_key_response():
    return {"insight": None, "error": "Groq API key not configured"}


@router.post("/bias-narrative")
async def bias_narrative(payload: dict = Body(...)):
    if not advisor.is_configured():
        return _missing_key_response()
    try:
        insight = await advisor.generate_bias_narrative(
            metrics=payload.get("metrics", {}),
            sensitive_col=payload.get("sensitive_col", ""),
            dataset_name=payload.get("dataset_name", "uploaded dataset"),
        )
        return {"insight": insight}
    except Exception:
        return {"insight": None, "error": "Failed to generate bias narrative"}


@router.post("/shap-insight")
async def shap_insight(payload: dict = Body(...)):
    if not advisor.is_configured():
        return _missing_key_response()
    try:
        insight = await advisor.explain_shap_features(
            shap_data=payload.get("shap_data", []),
            sensitive_col=payload.get("sensitive_col", ""),
        )
        return {"insight": insight}
    except Exception:
        return {"insight": None, "error": "Failed to generate SHAP insight"}


@router.post("/counterfactual-story")
async def counterfactual_story(payload: dict = Body(...)):
    if not advisor.is_configured():
        return _missing_key_response()
    try:
        insight = await advisor.generate_counterfactual_story(
            counterfactual_examples=payload.get("counterfactual_examples", []),
            sensitive_col=payload.get("sensitive_col", ""),
        )
        return {"insight": insight}
    except Exception:
        return {"insight": None, "error": "Failed to generate counterfactual story"}


@router.post("/mitigation-advice")
async def mitigation_advice(payload: dict = Body(...)):
    if not advisor.is_configured():
        return _missing_key_response()
    try:
        insight = await advisor.recommend_mitigation_strategy(
            simulation_results=payload.get("simulation_results", []),
            rl_recommendation=payload.get("rl_recommendation", ""),
            domain=payload.get("domain", "General"),
        )
        return {"insight": insight}
    except Exception:
        return {"insight": None, "error": "Failed to generate mitigation advice"}


@router.post("/intersectional-insight")
async def intersectional_insight(payload: dict = Body(...)):
    if not advisor.is_configured():
        return _missing_key_response()
    try:
        insight = await advisor.generate_intersectional_insight(
            intersectional_data=payload.get("intersectional_data", []),
        )
        return {"insight": insight}
    except Exception:
        return {"insight": None, "error": "Failed to generate intersectional insight"}


@router.post("/full-report")
async def full_report(payload: dict = Body(...)):
    if not advisor.is_configured():
        return _missing_key_response()
    try:
        insight = await advisor.generate_audit_report_narrative(full_audit_payload=payload)
        return {"insight": insight}
    except Exception:
        return {"insight": None, "error": "Failed to generate full report"}
