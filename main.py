from fastapi import FastAPI, HTTPException
import osmnx as ox
import networkx as nx
import random
from pydantic import BaseModel
from typing import List

app = FastAPI()

# --- 1. Models ---
class RouteRequest(BaseModel):
    lat: float
    lng: float
    distance_km: float

class RouteResponse(BaseModel):
    path: List[List[float]]
    actual_distance_km: float

# --- 2. Graph Helper (Tối ưu) ---
def get_graph_data(lat: float, lng: float, dist: int = 1000): # GIẢM bán kính xuống 1km
    # simplify=True giúp giảm số lượng node/edge, nhẹ RAM hơn
    G = ox.graph_from_point((lat, lng), dist=dist, network_type='walk', simplify=True)
    G_undirected = G.to_undirected()
    return G_undirected

# --- 3. Hàm Tính toán ---
def calculate_path_length(G, path: List) -> float:
    total_length = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        try:
            edge_data = G.get_edge_data(u, v)[0]
            total_length += edge_data.get('length', 0)
        except (KeyError, IndexError):
            continue
    return total_length

# --- 4. Hàm Chuyển đổi (Bám đường) ---
def convert_path_to_coords(G, path: List) -> List[List[float]]:
    detailed_coords = []
    if not path: return []
    
    # Thêm điểm đầu
    start_node_data = G.nodes[path[0]]
    detailed_coords.append([start_node_data['x'], start_node_data['y']])
    
    for i in range(len(path) - 1):
        node_u = path[i]
        node_v = path[i+1]
        try:
            edge_data = G.get_edge_data(node_u, node_v)[0]
            if 'geometry' in edge_data:
                linestring = edge_data['geometry']
                segment_coords = list(linestring.coords)
                # Kiểm tra chiều
                node_u_data = G.nodes[node_u]
                start_geom = segment_coords[0]
                end_geom = segment_coords[-1]
                dist_start = (start_geom[0]-node_u_data['x'])**2 + (start_geom[1]-node_u_data['y'])**2
                dist_end = (end_geom[0]-node_u_data['x'])**2 + (end_geom[1]-node_u_data['y'])**2
                
                if dist_end < dist_start:
                    segment_coords = list(reversed(segment_coords))
                
                if i > 0: segment_coords = segment_coords[1:]
                detailed_coords.extend([[c[0], c[1]] for c in segment_coords])
            else:
                node_v_data = G.nodes[node_v]
                detailed_coords.append([node_v_data['x'], node_v_data['y']])
        except: continue
            
    return detailed_coords

# --- 5. AI Logic (Smart Walk - Tối ưu) ---
def create_smart_walk(G, start_node, target_length, max_nodes=30): # GIẢM max_nodes
    path = [start_node]
    current_node = start_node
    current_length = 0
    
    for _ in range(max_nodes):
        if current_length > target_length: break
        neighbors = list(G.neighbors(current_node))
        if not neighbors: break

        possible_moves = []
        weights = [] 

        for neighbor in neighbors:
            if len(path) > 1 and neighbor == path[-2]:
                weights.append(0.01) 
            elif G.degree(neighbor) == 1 and neighbor != start_node:
                weights.append(0.1)
            elif neighbor in path:
                weights.append(0.1) 
            else:
                weights.append(1.0) 

            possible_moves.append(neighbor)
        
        if not possible_moves or sum(weights) == 0: break
        
        next_node = random.choices(possible_moves, weights=weights, k=1)[0]
        edge_len = G.get_edge_data(current_node, next_node)[0].get('length', 0)
        current_length += edge_len
        path.append(next_node)
        current_node = next_node
        
    return path

def find_best_loop_route(G, start_node, target_distance_meters, num_iterations=20): # GIẢM xuống 20 lần thử
    best_path = []
    best_fitness = -1
    best_length = 0
    
    walk_target = target_distance_meters * 0.6 

    for i in range(num_iterations):
        path_out = create_smart_walk(G, start_node, walk_target, max_nodes=40)
        if len(path_out) < 2: continue
        
        end_node = path_out[-1]
        try:
            # Dùng weight='length' để tìm đường ngắn nhất về
            return_path = nx.shortest_path(G, end_node, start_node, weight='length')
            full_path = path_out + return_path[1:]
            
            total_len = calculate_path_length(G, full_path)
            
            # Fitness: Ưu tiên độ dài gần đúng
            diff = abs(total_len - target_distance_meters)
            fitness = 1.0 / (diff + 1.0)

            if fitness > best_fitness:
                best_fitness = fitness
                best_path = full_path
                best_length = total_len
                
        except: continue

    return best_path, best_length

# --- 6. API Endpoint ---
@app.post("/api/v1/generate_route", response_model=RouteResponse)
def generate_route(request: RouteRequest):
    try:
        # 1. Tải đồ thị (Giảm bán kính xuống 1km để nhẹ RAM)
        G = get_graph_data(request.lat, request.lng, dist=1200)
        
        # 2. Tìm node bắt đầu
        # Sửa lỗi cú pháp osmnx mới nhất: dùng danh sách [lng], [lat]
        start_node = ox.nearest_nodes(G, [request.lng], [request.lat])[0]

        # 3. Chạy thuật toán (Giảm số lần lặp xuống 20)
        path_nodes, length = find_best_loop_route(
            G, start_node, 
            request.distance_km * 1000, 
            num_iterations=20 # <-- QUAN TRỌNG: Giảm từ 100 xuống 20
        )
        
        if not path_nodes:
             raise HTTPException(status_code=404, detail="Không tìm thấy đường chạy.")

        coords = convert_path_to_coords(G, path_nodes)
        
        return RouteResponse(path=coords, actual_distance_km=round(length/1000, 2))

    except Exception as e:
        print(f"LỖI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "WeRun AI Backend is running!"}