# main.py (Phiên bản GNN Inference)
from fastapi import FastAPI, HTTPException
import osmnx as ox
import networkx as nx
import random
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import os
from math import sqrt

app = FastAPI()

# --- CẤU HÌNH ---
MODEL_PATH = "gnn_model.pth"

# --- 1. ĐỊNH NGHĨA MODEL GNN (Phải giống hệt lúc Train) ---
class RouteGNN(nn.Module):
    def __init__(self):
        super(RouteGNN, self).__init__()
        self.conv1 = SAGEConv(3, 256)
        self.conv2 = SAGEConv(256, 256)
        self.conv3 = SAGEConv(256, 256)
        
        self.edge_pred = nn.Sequential(
            nn.Linear(514, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_attr, target_dist, batch=None):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()
        
        row, col = edge_index
        target_dist_exp = target_dist.repeat(row.size(0), 1)
        
        edge_feat = torch.cat([h[row], h[col], edge_attr, target_dist_exp], dim=1)
        return self.edge_pred(edge_feat).squeeze()

# --- 2. LOAD MODEL ---
device = torch.device('cpu') # Inference dùng CPU
gnn_model = None

if os.path.exists(MODEL_PATH):
    try:
        gnn_model = RouteGNN().to(device)
        # Load weights, map_location để đảm bảo chạy được trên CPU dù train trên GPU
        gnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        gnn_model.eval()
        print(f"✅ Đã load GNN Model: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
else:
    print("⚠️ Không tìm thấy file model, sẽ chạy chế độ heuristic cũ.")

# --- 3. CÁC HÀM HELPER ---
class RouteRequest(BaseModel):
    lat: float; lng: float; distance_km: float

class RouteResponse(BaseModel):
    path: List[List[float]]; actual_distance_km: float

def get_graph_data(lat, lng, dist=2000):
    G = ox.graph_from_point((lat, lng), dist=dist, network_type='walk', simplify=True)
    return G.to_undirected()

def calculate_path_length(G, path):
    total = 0
    for i in range(len(path) - 1):
        try:
            total += G.get_edge_data(path[i], path[i+1])[0].get('length', 0)
        except: continue
    return total

# --- HÀM CHẤM ĐIỂM BẢN ĐỒ BẰNG GNN ---
def score_graph_with_gnn(G, target_m, start_node):
    if gnn_model is None: return G

    # Preprocess Graph -> Tensor
    node_ids = list(G.nodes())
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    
    lats = [G.nodes[n]['y'] for n in node_ids]
    lngs = [G.nodes[n]['x'] for n in node_ids]
    min_lat, min_lng = min(lats), min(lngs)
    
    x = []
    for nid in node_ids:
        is_start = 1.0 if nid == start_node else 0.0
        x.append([
            (G.nodes[nid]['y'] - min_lat) * 10000,
            (G.nodes[nid]['x'] - min_lng) * 10000,
            is_start
        ])
    x = torch.tensor(x, dtype=torch.float)
    
    edge_index = []
    edge_attr = []
    edges_list = []
    
    for u, v, data in G.edges(data=True):
        if u in node_map and v in node_map:
            u_idx, v_idx = node_map[u], node_map[v]
            edge_index.append([u_idx, v_idx])
            
            length = data.get('length', 0) / 100.0
            edge_attr.append([length])
            edges_list.append((u, v))
            
    if not edge_index: return G

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    target_dist = torch.tensor([target_m / 1000.0], dtype=torch.float)
    
    # Inference
    with torch.no_grad():
        scores = gnn_model(x, edge_index, edge_attr, target_dist, None)
        probs = torch.sigmoid(scores).numpy()
    
    # Gán điểm ngược lại vào Graph
    for i, (u, v) in enumerate(edges_list):
        score = float(probs[i])
        # Lưu score vào thuộc tính edge
        if G.has_edge(u, v):
            G[u][v][0]['gnn_score'] = score
            
    return G

# --- 4. AI LOGIC MỚI (Dùng GNN Score) ---
def create_gnn_walk(G, start_node, target_length, max_nodes=60):
    path = [start_node]; curr = start_node; curr_len = 0
    
    for _ in range(max_nodes):
        if curr_len >= target_length: break
        neighbors = list(G.neighbors(curr))
        if not neighbors: break
        
        candidates = []; weights = []
        for n in neighbors:
            # Lấy điểm GNN (mặc định 0.5 nếu không có)
            try:
                base_score = G[curr][n][0].get('gnn_score', 0.5)
            except: base_score = 0.5

            # Kết hợp luật heuristic cũ
            if len(path) > 1 and n == path[-2]: base_score *= 0.01
            elif n in path: base_score *= 0.2
            
            # Tăng cường độ mạnh của GNN (bình phương)
            final_weight = base_score ** 2
            
            candidates.append(n)
            weights.append(final_weight)
            
        if sum(weights) == 0: weights = [1.0] * len(candidates)
        
        next_node = random.choices(candidates, weights=weights, k=1)[0]
        try:
            l = G.get_edge_data(curr, next_node)[0].get('length', 0)
            curr_len += l
            path.append(next_node); curr = next_node
        except: break
            
    return path

def find_best_route(G, start_node, target_m):
    # B1: Chấm điểm bản đồ bằng GNN
    G = score_graph_with_gnn(G, target_m, start_node)
    
    best_path = []; best_score = -1; best_len = 0
    
    # B2: Chạy tìm kiếm (ít lần hơn vì đã có GNN hướng dẫn)
    for i in range(20):
        path_out = create_gnn_walk(G, start_node, target_m * 0.55)
        if len(path_out) < 3: continue
        
        last = path_out[-1]
        try:
            path_ret = nx.shortest_path(G, last, start_node, weight='length')
            full = path_out + path_ret[1:]
            length = calculate_path_length(G, full)
            
            diff = abs(length - target_m)
            score = 1.0 / (diff + 1.0)
            
            if score > best_score:
                best_score = score; best_path = full; best_len = length
        except: continue
        
    # Fallback Out-and-Back nếu thất bại
    if not best_path or abs(best_len - target_m) > target_m * 0.3:
        try:
            rad = target_m / 2.0
            lens = nx.single_source_dijkstra_path_length(G, start_node, cutoff=rad*1.2, weight='length')
            best_n = None; min_d = float('inf')
            for n, d in lens.items():
                diff = abs(d - rad)
                if diff < min_d: min_d = diff; best_n = n
            if best_n:
                p_out = nx.shortest_path(G, start_node, best_n, weight='length')
                return p_out + p_out[::-1][1:], lens[best_n]*2
        except: pass

    return best_path, best_len

# --- 5. CONVERT COORDS (Giữ nguyên) ---
def convert_path_to_coords(G, path):
    detailed_coords = []
    if not path: return []
    start_node = G.nodes[path[0]]
    detailed_coords.append([start_node['x'], start_node['y']])
    
    for i in range(len(path) - 1):
        u = path[i]; v = path[i+1]
        try:
            data = G.get_edge_data(u, v)[0]
            if 'geometry' in data:
                coords = list(data['geometry'].coords)
                u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
                start, end = coords[0], coords[-1]
                d_start = (start[0]-u_x)**2 + (start[1]-u_y)**2
                d_end = (end[0]-u_x)**2 + (end[1]-u_y)**2
                if d_end < d_start: coords = list(reversed(coords))
                detailed_coords.extend([[c[0], c[1]] for c in coords[1:]])
            else:
                node = G.nodes[v]
                detailed_coords.append([node['x'], node['y']])
        except: continue
    return detailed_coords

# --- API ---
@app.post("/api/v1/generate_route", response_model=RouteResponse)
def generate_route(request: RouteRequest):
    try:
        G = get_graph_data(request.lat, request.lng)
        start_node = ox.nearest_nodes(G, [request.lng], [request.lat])[0]
        
        path_nodes, length = find_best_route(G, start_node, request.distance_km * 1000)
        
        if not path_nodes:
             raise HTTPException(status_code=404, detail="Không tìm thấy đường.")

        coords = convert_path_to_coords(G, path_nodes)
        return RouteResponse(path=coords, actual_distance_km=round(length/1000, 2))

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "WeRun AI Backend (GNN Powered) is Running!"}