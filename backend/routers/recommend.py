from fastapi import APIRouter, HTTPException
from services.rl_agent import RLAgent
from models.schemas import RecommendationRequest, RecommendationResponse

router = APIRouter(prefix="/api")
agent = RLAgent()

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_strategy(request: RecommendationRequest):
    try:
        recommendation = agent.get_recommendation(
            bias_score=request.bias_score,
            accuracy=request.accuracy
        )
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
