# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import osmnx as ox
import networkx as nx
import random
from math import radians, cos, sin, asin, sqrt

app = FastAPI()

# -----------------------
# Pydantic models
# -----------------------
class RouteRequest(BaseModel):
    lat: float
    lng: float
    distance_km: float

class RouteResponse(BaseModel):
    path: List[List[float]]        # list of [lng, lat]
    actual_distance_km: float

# -----------------------
# Utilities
# -----------------------
def haversine(lon1, lat1, lon2, lat2):
    """
    Haversine distance in meters between two (lon, lat).
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    R = 6371000  # meters
    return R * c

# -----------------------
# Graph helper
# -----------------------
def get_graph_data(lat: float, lng: float, dist: int = 1200):
    """
    Tải graph xung quanh (lat, lng) với bán kính dist (m).
    Trả về graph undircted (dễ xử lý).
    """
    try:
        G = ox.graph_from_point((lat, lng), dist=dist, network_type='walk', simplify=True)
        G_undirected = G.to_undirected()
        return G_undirected
    except Exception as e:
        print(f"Error tải graph: {e}")
        raise

# -----------------------
# Path length calculator
# -----------------------
def calculate_path_length(G: nx.Graph, path: List[int]) -> float:
    """
    Tính tổng chiều dài (meters) của đường đi nối các node trong 'path'.
    Dùng 'length' trên edge nếu có, nếu không có thì fallback dùng Haversine theo tọa độ node.
    """
    if not path or len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        try:
            edge_data = G.get_edge_data(u, v)
            if not edge_data:
                # no edge info, fallback to haversine between node coords
                ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                total += haversine(ux, uy, vx, vy)
            else:
                # if multiple parallel edges, pick the first with 'length' if possible
                if isinstance(edge_data, dict):
                    # osmnx often stores {0: {...}, 1: {...}} for multigraphs
                    # take first sub-dict that contains 'length'
                    found = False
                    for k in edge_data:
                        e = edge_data[k]
                        if 'length' in e:
                            total += float(e.get('length', 0.0))
                            found = True
                            break
                    if not found:
                        # fallback to geometry/haversine
                        ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                        vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                        total += haversine(ux, uy, vx, vy)
                else:
                    # single edge dict
                    total += float(edge_data.get('length', 0.0))
        except Exception:
            # If anything goes wrong, fallback to node-to-node distance
            try:
                ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                total += haversine(ux, uy, vx, vy)
            except Exception:
                continue
    return total

# -----------------------
# Convert node path -> detailed coords (bám đường)
# -----------------------
def convert_path_to_coords(G: nx.Graph, path: List[int]) -> List[List[float]]:
    """
    Chuyển một danh sách node thành danh sách tọa độ [lng, lat] bám theo geometry của các edge nếu có.
    Giữ thứ tự đúng, tránh trùng lặp điểm.
    """
    if not path:
        return []

    coords = []
    # add start node
    try:
        start = G.nodes[path[0]]
        coords.append([start['x'], start['y']])
    except Exception:
        pass

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        try:
            edge_data = G.get_edge_data(u, v)
            chosen = None
            if edge_data:
                # edge_data can be a multi-dict; pick first dict that has geometry or length
                if isinstance(edge_data, dict):
                    for k in edge_data:
                        candidate = edge_data[k]
                        if 'geometry' in candidate or 'length' in candidate:
                            chosen = candidate
                            break
                else:
                    chosen = edge_data

            if chosen and 'geometry' in chosen:
                line = chosen['geometry']
                segment_coords = list(line.coords)
                # ensure segment orientation matches u -> v: check which end is closer to node u
                ucoord = (G.nodes[u]['x'], G.nodes[u]['y'])
                start_geom = segment_coords[0]
                end_geom = segment_coords[-1]
                dist_start = (start_geom[0] - ucoord[0])**2 + (start_geom[1] - ucoord[1])**2
                dist_end = (end_geom[0] - ucoord[0])**2 + (end_geom[1] - ucoord[1])**2
                if dist_end < dist_start:
                    segment_coords = list(reversed(segment_coords))
                # to avoid duplicate point, skip first if coords already endswith same
                if coords and len(coords) > 0 and coords[-1] == [segment_coords[0][0], segment_coords[0][1]]:
                    segment_coords = segment_coords[1:]
                coords.extend([[c[0], c[1]] for c in segment_coords])
            else:
                # no geometry -> just append v coordinate (avoid duplicate)
                vdata = G.nodes.get(v)
                if vdata:
                    if not coords or coords[-1] != [vdata['x'], vdata['y']]:
                        coords.append([vdata['x'], vdata['y']])
        except Exception as e:
            # skip problematic edge but continue
            print(f"Warning convert edge {u}->{v}: {e}")
            continue

    return coords

# -----------------------
# Smart walk (randomized walk biased by rules)
# -----------------------
def create_smart_walk(G: nx.Graph, start_node: int, target_length: float, max_nodes: int = 30) -> List[int]:
    """
    Sinh 1 đường ngẫu nhiên bắt đầu từ start_node, cố gắng gần target_length (meters).
    Trả về danh sách node theo thứ tự.
    """
    path = [start_node]
    current = start_node
    current_len = 0.0

    for _ in range(max_nodes):
        if current_len >= target_length:
            break
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break

        possible = []
        weights = []
        for nb in neighbors:
            # don't allow immediate backtracking to previous node easily
            if len(path) > 1 and nb == path[-2]:
                w = 0.02
            elif G.degree(nb) == 1 and nb != start_node:
                # dead end - discourage
                w = 0.05
            elif nb in path:
                # already visited - discourage but allow
                w = 0.1
            else:
                w = 1.0
            possible.append(nb)
            weights.append(w)

        if not possible or sum(weights) == 0:
            break

        next_node = random.choices(possible, weights=weights, k=1)[0]
        # edge length
        try:
            e = G.get_edge_data(current, next_node)
            edge_len = None
            if e:
                if isinstance(e, dict):
                    # take first sub-edge that has length
                    for k in e:
                        if 'length' in e[k]:
                            edge_len = float(e[k]['length'])
                            break
                else:
                    edge_len = float(e.get('length', 0.0))
            if edge_len is None:
                ux, uy = G.nodes[current]['x'], G.nodes[current]['y']
                vx, vy = G.nodes[next_node]['x'], G.nodes[next_node]['y']
                edge_len = haversine(ux, uy, vx, vy)
        except Exception:
            # fallback
            try:
                ux, uy = G.nodes[current]['x'], G.nodes[current]['y']
                vx, vy = G.nodes[next_node]['x'], G.nodes[next_node]['y']
                edge_len = haversine(ux, uy, vx, vy)
            except Exception:
                edge_len = 0.0

        current_len += edge_len
        path.append(next_node)
        current = next_node

    return path

# -----------------------
# Find best loop route
# -----------------------
def find_best_loop_route(G: nx.Graph, start_node: int, target_distance_meters: float, num_iterations: int = 20):
    """
    Thực hiện nhiều lần random walk + nối đường ngắn nhất về start để tạo loop.
    Trả về best_path (node list) và best_length (meters).
    """
    best_path = []
    best_len = 0.0
    best_fitness = -1.0

    for i in range(max(1, num_iterations)):
        # choose a walk target fraction of target distance (randomize a bit)
        walk_target = target_distance_meters * random.uniform(0.4, 0.75)
        walk = create_smart_walk(G, start_node, walk_target, max_nodes=40)
        if len(walk) < 2:
            continue
        end_node = walk[-1]

        # if ended at start, it's already a loop
        if end_node == start_node:
            full = walk
            total_len = calculate_path_length(G, full)
        else:
            # try to find shortest path back
            try:
                return_path = nx.shortest_path(G, source=end_node, target=start_node, weight='length')
                full = walk + return_path[1:]
                total_len = calculate_path_length(G, full)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                full = walk
                total_len = calculate_path_length(G, full)

        diff = abs(total_len - target_distance_meters)
        fitness = 1.0 / (diff + 1.0)

        if fitness > best_fitness:
            best_fitness = fitness
            best_len = total_len
            best_path = full

    return best_path, best_len

# -----------------------
# API endpoint
# -----------------------
@app.post("/api/v1/generate_route", response_model=RouteResponse)
def generate_route(request: RouteRequest):
    target_m = request.distance_km * 1000.0
    try:
        # 1. tải graph (khoảng 1.2 km mặc định, có thể chỉnh nếu cần)
        G = get_graph_data(request.lat, request.lng, dist=1200)

        # 2. tìm node gần điểm bắt đầu
        try:
            # ox.nearest_nodes signature may vary by version; many accept (G, X, Y)
            start_node = ox.nearest_nodes(G, request.lng, request.lat)
        except Exception:
            # fallback using distance module
            start_node = ox.distance.nearest_nodes(G, X=[request.lng], Y=[request.lat])[0]

        # 3. tìm đường tốt nhất
        path_nodes, path_len = find_best_loop_route(G, start_node, target_m, num_iterations=20)

        if not path_nodes or path_len < (target_m * 0.1):
            raise HTTPException(status_code=404, detail="Không thể tìm thấy đường chạy phù hợp tại khu vực này.")

        coords = convert_path_to_coords(G, path_nodes)
        if not coords:
            # fallback: convert nodes to simple coords
            coords = []
            for n in path_nodes:
                nd = G.nodes.get(n)
                if nd:
                    coords.append([nd['x'], nd['y']])

        return RouteResponse(path=coords, actual_distance_km=round(path_len / 1000.0, 3))

    except HTTPException:
        raise
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ nội bộ: {e}")

@app.get("/")
def read_root():
    return {"status": "ok", "note": "Route generator service is running."}
