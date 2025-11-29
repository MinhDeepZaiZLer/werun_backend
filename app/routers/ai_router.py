from fastapi import APIRouter, HTTPException
from app.models.schemas import RouteRequest, RouteResponse
from app.services.ai_service import (
    get_graph_data, find_best_route, convert_path_to_coords
)
import osmnx as ox

router = APIRouter(prefix="/api/v1")

@router.post("/generate_route", response_model=RouteResponse)
def generate_route(request: RouteRequest):
    try:
        G = get_graph_data(request.lat, request.lng)
        start_node = ox.nearest_nodes(G, [request.lng], [request.lat])[0]

        nodes, length = find_best_route(G, start_node, request.distance_km*1000)
        coords = convert_path_to_coords(G, nodes)

        return RouteResponse(
            path=coords,
            actual_distance_km=round(length/1000, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
