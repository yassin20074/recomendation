from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app import get_smart_recommendation

app = FastAPI(
    title="AI Recommendation system ",
    description="Production-grade Microservice built with FastAPI and SentenceTransformers"
)

 
class ItineraryRequest(BaseModel):
    user_id: int
    budget: float
    city: str
    preference: str   

@app.post("/api/recommend/itinerary")
def get_recommendation_endpoint(request: ItineraryRequest):
  
    program, total_cost, status = get_smart_recommendation(
        city=request.city,
        preference=request.preference,
        budget=request.budget
    )
    
    
    if status == "CITY_NOT_FOUND":
        raise HTTPException(status_code=404, detail="عفواً، لا توجد أماكن مغطاة في هذه المدينة حالياً.")
        
    if status == "BUDGET_TOO_LOW":
        raise HTTPException(status_code=400, detail="الميزانية المحددة ضئيلة جداً لعمل برنامج يومي.")

 
    return {
        "title": f"يوم {request.preference} في {request.city}",
        "summary": f"6 ساعات | {int(total_cost)} جنيه",
        "program": program
    }