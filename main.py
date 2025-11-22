from fastapi import FastAPI, HTTPException
import osmnx as ox
import networkx as nx
import random
from pydantic import BaseModel
from typing import List
from math import sqrt

app = FastAPI()

# --- 1. Models ---
class RouteRequest(BaseModel):
    lat: float
    lng: float
    distance_km: float

class RouteResponse(BaseModel):
    path: List[List[float]]
    actual_distance_km: float

# --- 2. Graph Helper ---
def get_graph_data(lat: float, lng: float, dist: int = 1200):
    # Tải đồ thị, giữ lại geometry chi tiết
    G = ox.graph_from_point((lat, lng), dist=dist, network_type='walk', simplify=True)
    # Chuyển sang vô hướng để dễ tìm đường 2 chiều
    G_undirected = G.to_undirected()
    return G_undirected

# --- 3. Hàm Tính toán ---
def calculate_path_length(G, path: List) -> float:
    total_length = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        try:
            # Lấy dữ liệu cạnh đầu tiên (key=0)
            edge_data = G.get_edge_data(u, v)[0]
            total_length += edge_data.get('length', 0)
        except:
            continue
    return total_length

# --- 4. Hàm Chuyển đổi (SỬA LỖI XUYÊN ĐƯỜNG) ---
def convert_path_to_coords(G, path: List) -> List[List[float]]:
    detailed_coords = []
    if not path: return []
    
    # Thêm điểm đầu tiên
    start_node = G.nodes[path[0]]
    detailed_coords.append([start_node['x'], start_node['y']])
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        try:
            edge_data = G.get_edge_data(u, v)[0]
            
            # KIỂM TRA GEOMETRY (Hình dáng đường)
            if 'geometry' in edge_data:
                linestring = edge_data['geometry']
                # Lấy danh sách tọa độ của đoạn đường cong
                segment_coords = list(linestring.coords) # [(lng, lat), ...]
                
                # XỬ LÝ CHIỀU: Kiểm tra xem geometry đang lưu xuôi hay ngược
                u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
                
                # Lấy điểm đầu và cuối của geometry
                start_geom = segment_coords[0]
                end_geom = segment_coords[-1]
                
                # Tính khoảng cách từ Node U đến điểm đầu/cuối của geometry
                dist_to_start = (start_geom[0] - u_x)**2 + (start_geom[1] - u_y)**2
                dist_to_end = (end_geom[0] - u_x)**2 + (end_geom[1] - u_y)**2
                
                # Nếu Node U gần điểm cuối hơn -> Geometry đang bị ngược -> Đảo chiều
                if dist_to_end < dist_to_start:
                    segment_coords = list(reversed(segment_coords))
                
                # Thêm các điểm trung gian (bỏ điểm đầu để tránh trùng)
                detailed_coords.extend([[p[0], p[1]] for p in segment_coords[1:]])
            
            else:
                # Nếu là đường thẳng (không có geometry), chỉ thêm điểm đích
                v_data = G.nodes[v]
                detailed_coords.append([v_data['x'], v_data['y']])
                
        except Exception as e:
            print(f"Lỗi cạnh {u}-{v}: {e}")
            # Fallback: Nối thẳng nếu lỗi
            v_data = G.nodes[v]
            detailed_coords.append([v_data['x'], v_data['y']])
            
    return detailed_coords

# --- 5. AI Logic (SMART WALK - Tránh chạy lui) ---
def create_smart_walk(G, start_node, target_length, max_nodes=40):
    path = [start_node]
    curr = start_node
    curr_len = 0
    
    for _ in range(max_nodes):
        if curr_len >= target_length: break
        
        neighbors = list(G.neighbors(curr))
        if not neighbors: break
        
        # --- HỆ THỐNG TÍNH ĐIỂM ---
        candidates = []
        weights = []
        
        for n in neighbors:
            score = 1.0
            # 1. Phạt nặng nếu quay đầu ngay lập tức (Node vừa đi qua)
            if len(path) > 1 and n == path[-2]:
                score = 0.01 
            # 2. Phạt nếu đi vào đường cụt (chỉ có 1 lối ra)
            elif G.degree(n) == 1:
                score = 0.1
            # 3. Phạt nếu đi lại đường cũ (đã có trong path)
            elif n in path:
                score = 0.2
            
            candidates.append(n)
            weights.append(score)
            
        # Nếu kẹt (tất cả điểm đều thấp), buộc phải quay đầu
        if sum(weights) == 0: break
        
        # Chọn bước tiếp theo dựa trên điểm số
        next_node = random.choices(candidates, weights=weights, k=1)[0]
        
        try:
            edge_len = G.get_edge_data(curr, next_node)[0].get('length', 0)
            curr_len += edge_len
            path.append(next_node)
            curr = next_node
        except:
            break
            
    return path

def find_best_loop(G, start_node, target_dist, iterations=30):
    best_path = []
    best_score = -1
    best_len = 0
    
    # Mục tiêu: Đi xa khoảng 60%, sau đó tìm đường về
    walk_dist = target_dist * 0.6
    
    for _ in range(iterations):
        # 1. Đi bộ thông minh
        path_out = create_smart_walk(G, start_node, walk_dist)
        if len(path_out) < 3: continue
        
        # 2. Tìm đường về ngắn nhất (Dijkstra)
        last_node = path_out[-1]
        try:
            path_return = nx.shortest_path(G, last_node, start_node, weight='length')
            full_path = path_out + path_return[1:] # Nối lại
            
            total_len = calculate_path_length(G, full_path)
            
            # 3. Chấm điểm: Càng gần mục tiêu càng tốt
            diff = abs(total_len - target_dist)
            score = 1.0 / (diff + 1.0)
            
            if score > best_score:
                best_score = score
                best_path = full_path
                best_len = total_len
        except:
            continue
            
    return best_path, best_len

# --- 6. API Endpoint ---
@app.post("/api/v1/generate_route", response_model=RouteResponse)
def generate_route(request: RouteRequest):
    try:
        # 1. Tăng bán kính tìm kiếm lên 2km (để dễ tìm đường 5km hơn)
        G = get_graph_data(request.lat, request.lng, dist=2000) 
        
        # ... (code tìm start_node giữ nguyên) ...
        start_node = ox.nearest_nodes(G, [request.lng], [request.lat])[0]

        # 2. Tăng số lần thử lên 30 (nếu server chịu được)
        path_nodes, length = find_best_loop_route(
            G, 
            start_node, 
            request.distance_km * 1000, 
            num_iterations=30, 
            max_nodes_per_walk=50 # Tăng khả năng đi xa hơn
        )
        
        if not path_nodes:
             # Đây là lỗi dự kiến, trả về 404
             raise HTTPException(status_code=404, detail="Không tìm thấy đường chạy phù hợp ở khu vực này. Hãy thử giảm quãng đường hoặc chọn vị trí khác.")

        coords = convert_path_to_coords(G, path_nodes)
        
        return RouteResponse(path=coords, actual_distance_km=round(length/1000, 2))

    # === SỬA ĐOẠN NÀY ===
    except HTTPException as he:
        raise he # Nếu là lỗi HTTP (như 404 ở trên), ném ra ngay, không bọc lại
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi Server: {str(e)}")