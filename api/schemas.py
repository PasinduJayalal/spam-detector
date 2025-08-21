from pydantic import BaseModel


class PredictIn(BaseModel):
    model : str
    text: str | None = None
    texts : list[str] | None = None
    
class PredictOut(BaseModel):
    label: str
    score: float | None = None
    model : str 
    text: str | None = None

class PredictBatchOut(BaseModel):
    results: list[PredictOut]
