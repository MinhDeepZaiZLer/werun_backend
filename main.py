# main.py
from fastapi import FastAPI, HTTPException
import osmnx as ox
import networkx as nx
import random
from pydantic import BaseModel
from typing import List, Optional
# Thêm thư viện tính khoảng cách (Euclidean)
from math import sqrt 

app = FastAPI()

# --- 1. Pydantic Models (Input/Output) ---
class RouteRequest(BaseModel):
    lat: float
    lng: float
    distance_km: float

class RouteResponse(BaseModel):
    path: List[List[float]] # List of [lng, lat]
    actual_distance_km: float

# --- 2. Graph Helper ---
def get_graph_data(lat: float, lng: float, dist: int = 1500):
    print(f"Đang tải graph tại ({lat}, {lng}) với bán kính {dist}m...")
    G = ox.graph_from_point((lat, lng), dist=dist, network_type='walk')
    G_undirected = G.to_undirected()
    print("Graph đã tải xong.")
    return G_undirected

# --- 3. HÀM TÍNH TOÁN (Hàm phụ) ---
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

# --- 4. HÀM CHUYỂN ĐỔI (Sửa lỗi "xuyên đường" - RẤT QUAN TRỌNG) ---
def convert_path_to_coords(G, path: List) -> List[List[float]]:
    """
    Chuyển đổi một danh sách các "node mốc" thành tọa độ [lng, lat] chi tiết,
    ĐẢM BẢO BÁM ĐƯỜNG và ĐÚNG THỨ TỰ.
    """
    detailed_coords = []
    if not path:
        return []
        
    # Lặp qua từng cặp node trong đường chạy (ví dụ: A -> B, B -> C)
    for i in range(len(path) - 1):
        node_u_id = path[i]
        node_v_id = path[i+1]
        
        node_u_data = G.nodes[node_u_id]
        
        try:
            # Lấy dữ liệu của cạnh (edge) giữa 2 node
            edge_data = G.get_edge_data(node_u_id, node_v_id)[0]
            
            if 'geometry' in edge_data:
                linestring = edge_data['geometry']
                segment_coords = list(linestring.coords)
                
                # --- LOGIC KIỂM TRA THỨ TỰ (Sửa lỗi) ---
                # Lấy tọa độ điểm đầu và cuối của geometry
                start_geom = segment_coords[0]
                end_geom = segment_coords[-1]
                
                # Lấy tọa độ của node_u
                u_coord = (node_u_data['x'], node_u_data['y']) # (lng, lat)

                # Tính khoảng cách
                dist_to_start = sqrt((start_geom[0] - u_coord[0])**2 + (start_geom[1] - u_coord[1])**2)
                dist_to_end = sqrt((end_geom[0] - u_coord[0])**2 + (end_geom[1] - u_coord[1])**2)

                # Nếu điểm cuối của geometry gần node_u hơn
                # -> Geometry đang bị lưu ngược (V -> U)
                if dist_to_end < dist_to_start:
                    segment_coords = list(reversed(segment_coords))
                # --- KẾT THÚC SỬA LỖI ---

                # Bỏ điểm đầu tiên (nếu không phải là điểm bắt đầu)
                # để tránh trùng lặp
                if i > 0:
                    segment_coords = segment_coords[1:]
                    
                detailed_coords.extend([[coord[0], coord[1]] for coord in segment_coords])
            
            else:
                # Nếu cạnh không có geometry (đường thẳng)
                node_v_data = G.nodes[node_v_id]
                
                # Chỉ thêm điểm đầu nếu là điểm bắt đầu
                if i == 0:
                    detailed_coords.append([node_u_data['x'], node_u_data['y']])
                
                detailed_coords.append([node_v_data['x'], node_v_data['y']])
                
        except Exception as e:
            print(f"Bỏ qua cạnh không hợp lệ: {node_u_id} -> {node_v_id}. Lỗi: {e}")
            continue
            
    return detailed_coords

# --- 5. HÀM AI (Smart Walk - Giữ nguyên) ---
def create_smart_walk(G, start_node, target_length, max_nodes=50):
    path = [start_node]
    current_node = start_node
    current_length = 0
    
    for _ in range(max_nodes):
        if current_length > target_length:
            break 
            
        neighbors = list(G.neighbors(current_node))
        if not neighbors: break

        possible_moves = []
        weights = [] 

        for neighbor in neighbors:
            if len(path) > 1 and neighbor == path[-2]:
                weights.append(0.01) # Cấm quay đầu
            elif G.degree(neighbor) == 1 and neighbor != start_node:
                weights.append(0.1) # Hạn chế ngõ cụt
            elif neighbor in path:
                weights.append(0.2) # Hạn chế đi lại
            else:
                weights.append(1.0) # Khuyến khích

            possible_moves.append(neighbor)
        
        if all(w < 0.2 for w in weights):
            weights = [1.0 if w >= 0.1 else 0.01 for w in weights] # Cho phép đi lại (trừ quay đầu)
        
        if not possible_moves or sum(weights) == 0:
            break # Bị kẹt
            
        next_node = random.choices(possible_moves, weights=weights, k=1)[0]
        
        current_length += G.get_edge_data(current_node, next_node)[0].get('length', 0)
        path.append(next_node)
        current_node = next_node
        
    return path

# --- 6. HÀM TÌM KIẾM (Đã nâng cấp) ---
def find_best_loop_route(G, start_node, target_distance_meters, num_iterations=100):
    print(f"Bắt đầu tìm đường chạy... Target: {target_distance_meters}m")
    best_path = []
    best_fitness = -1
    best_length = 0
    
    for i in range(num_iterations):
        # Mục tiêu độ dài đường đi = 50-70% tổng quãng đường
        walk_target_length = target_distance_meters * random.uniform(0.5, 0.7)
        
        path_out = create_smart_walk(G, start_node, walk_target_length, max_nodes=50)
        
        if len(path_out) < 2: continue
        end_node = path_out[-1]
        
        # Nếu đi bộ đã đủ xa, thử dùng luôn (nếu nó là vòng lặp)
        if end_node == start_node:
             full_path = path_out
             total_length = calculate_path_length(G, full_path)
        else:
            # Tìm đường "về" ngắn nhất (Dijkstra)
            try:
                return_path_nodes = nx.shortest_path(G, source=end_node, target=start_node, weight='length')
                return_length = calculate_path_length(G, return_path_nodes)
                
                full_path = path_out + return_path_nodes[1:]
                total_length = calculate_path_length(G, path_out) + return_length
                
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                full_path = path_out
                total_length = calculate_path_length(G, path_out)

        # Đánh giá (Fitness) - Chỉ dựa trên độ dài
        difference = abs(total_length - target_distance_meters)
        fitness = 1.0 / (difference + 1.0) 

        if fitness > best_fitness:
            best_fitness = fitness
            best_path = full_path
            best_length = total_length

    print(f"Tìm kiếm hoàn tất. Đường chạy tốt nhất: {best_length:.2f}m")
    return best_path, best_length

# === 7. ENDPOINT CHÍNH (Giữ nguyên) ===
@app.post("/api/v1/generate_route", response_model=RouteResponse)
def generate_route(request: RouteRequest):
    target_distance = request.distance_km * 1000
    
    try:
        G = get_graph_data(request.lat, request.lng)
        start_node = ox.nearest_nodes(G, [request.lng], [request.lat])[0] 

        best_path_nodes, best_length = find_best_loop_route(
            G, 
            start_node, 
            target_distance,
            num_iterations=100
        )
        
        # Dùng hàm convert_path_to_coords MỚI (BÁM ĐƯỜNG)
        final_coords = convert_path_to_coords(G, best_path_nodes)
        
        if not final_coords or best_length < (target_distance * 0.1):
             raise HTTPException(status_code=404, detail="Không thể tìm thấy đường chạy phù hợp tại khu vực này.")

        return RouteResponse(path=final_coords, actual_distance_km=round(best_length / 1000, 2))

    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ: {e}")

@app.get("/")
def read_root():
    return {"message": "WeRun AI Backend is running!"}