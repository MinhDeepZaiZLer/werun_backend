from pydantic import BaseModel
from typing import List

class RouteRequest(BaseModel):
    lat: float
    lng: float
    distance_km: float

class RouteResponse(BaseModel):
    path: List[List[float]]
    actual_distance_km: float
