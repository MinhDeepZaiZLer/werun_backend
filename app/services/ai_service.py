import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import osmnx as ox
from math import sqrt
import random
import numpy as np
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "gnn_model.pth")

# =============================== MODEL =====================================
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

# =============================== LOAD MODEL =====================================
device = torch.device("cpu")
gnn_model = None

try:
    gnn_model = RouteGNN().to(device)
    gnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    gnn_model.eval()
    print(f"✅ Loaded GNN Model from {MODEL_PATH}")
except Exception as e:
    print("⚠️ Không thể load model GNN:", e)
    gnn_model = None

# =============================== GRAPH UTILS =====================================
def get_graph_data(lat, lng, dist=2000):
    G = ox.graph_from_point((lat, lng), dist=dist, network_type="walk", simplify=True)
    return G.to_undirected()

def calculate_path_length(G, path):
    total = 0
    for i in range(len(path) - 1):
        try:
            total += G.get_edge_data(path[i], path[i+1])[0].get("length", 0)
        except:
            continue
    return total

# =============================== SCORE GRAPH =====================================
def score_graph_with_gnn(G, target_m, start_node):
    if gnn_model is None:
        return G

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

    # Lưu raw edge data cẩn trọng (MultiGraph -> dict of keys)
    for u, v, raw in G.edges(data=True):
        try:
            # raw có thể là dict-of-keys (multi-edge), lấy 1 record phù hợp
            if isinstance(raw, dict) and (0 in raw or any(isinstance(k, int) for k in raw.keys())):
                # lấy giá trị đầu tiên
                first = next(iter(raw.values()))
                data = first if isinstance(first, dict) else raw
            else:
                data = raw
        except Exception:
            data = raw

        if u in node_map and v in node_map:
            edge_index.append([node_map[u], node_map[v]])
            length = data.get('length', 0) / 100.0
            edge_attr.append([length])
            edges_list.append((u, v))

    if not edge_index:
        return G

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    target_dist = torch.tensor([target_m / 1000.0], dtype=torch.float)

    with torch.no_grad():
        scores = gnn_model(x, edge_index, edge_attr, target_dist, None)
        probs = torch.sigmoid(scores).numpy()

    # Gán score vào graph (cẩn thận với multi-edge / directed)
    for i, (u, v) in enumerate(edges_list):
        score = float(probs[i])
        # nếu tồn tại multi-edges, cập nhật phần tử đầu tiên (key 0 hoặc first key)
        try:
            # networkx MultiGraph: G[u][v] là dict keys -> values
            uv = G[u][v]
            if isinstance(uv, dict):
                # cập nhật tất cả các keys để nhất quán
                for key in list(uv.keys()):
                    if isinstance(uv[key], dict):
                        uv[key]['gnn_score'] = score
                    else:
                        # fallback
                        try:
                            uv[key] = dict(uv[key])
                            uv[key]['gnn_score'] = score
                        except:
                            pass
            else:
                uv['gnn_score'] = score
        except Exception:
            # fallback: trực tiếp gán nếu đơn giản
            try:
                G[u][v]['gnn_score'] = score
            except:
                pass

        # nếu có cạnh ngược tồn tại (undirected stored twice), cập nhật ngược lại
        try:
            if G.has_edge(v, u):
                vv = G[v][u]
                if isinstance(vv, dict):
                    for key in list(vv.keys()):
                        if isinstance(vv[key], dict):
                            vv[key]['gnn_score'] = score
                else:
                    vv['gnn_score'] = score
        except:
            pass

    return G


# =============================== GENERATE WALK =====================================
def create_gnn_walk(G, start_node, target_length, max_nodes=60):
    path = [start_node]
    curr = start_node
    curr_len = 0

    for _ in range(max_nodes):
        if curr_len >= target_length:
            break

        neighbors = list(G.neighbors(curr))
        if not neighbors:
            break

        candidates = []
        weights = []

        for n in neighbors:
            # Lấy edge data an toàn (MultiGraph)
            try:
                edge_data_raw = G.get_edge_data(curr, n)
                # edge_data_raw có thể là dict of keys -> lấy first dict
                if isinstance(edge_data_raw, dict):
                    # lấy first value that is dict-like
                    val = next((v for v in edge_data_raw.values() if isinstance(v, dict)), None)
                    data = val if val is not None else list(edge_data_raw.values())[0]
                else:
                    data = edge_data_raw
            except Exception:
                data = None

            # Bỏ qua nếu không có length hợp lệ
            length = None
            if isinstance(data, dict):
                length = data.get('length', None)
            if not length or length <= 0:
                continue

            # Lấy điểm GNN, default 0.5
            try:
                base_score = data.get('gnn_score', 0.5)
            except:
                base_score = 0.5

            # heuristic tránh quay lại quá nhanh
            if len(path) > 1 and n == path[-2]:
                base_score *= 0.01
            elif n in path:
                base_score *= 0.2

            candidates.append((n, length))
            weights.append(base_score ** 2)

        if not candidates:
            break

        # normalise weights and choose
        if sum(weights) == 0:
            weights = [1.0] * len(candidates)

        idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        next_node, edge_len = candidates[idx]

        curr_len += edge_len
        path.append(next_node)
        curr = next_node

    return path


# =============================== FINAL SEARCH =====================================
def find_best_route(G, start_node, target_m):
    G = score_graph_with_gnn(G, target_m, start_node)

    best_path = []
    best_score = -1
    best_len = 0

    for _ in range(20):
        out = create_gnn_walk(G, start_node, target_m * 0.55)
        if len(out) < 3:
            continue

        last = out[-1]
        try:
            ret = nx.shortest_path(G, last, start_node, weight="length")
            full = out + ret[1:]
            length = calculate_path_length(G, full)

            diff = abs(length - target_m)
            score = 1 / (diff + 1)

            if score > best_score:
                best_score = score
                best_path = full
                best_len = length
        except:
            continue

    return best_path, best_len

# =============================== COORD CONVERT =====================================
def convert_path_to_coords(G, path):
    detailed_coords = []
    if not path:
        return []

    # append start node coords
    start_node = G.nodes[path[0]]
    detailed_coords.append([start_node['x'], start_node['y']])

    for i in range(len(path) - 1):
        u = path[i]; v = path[i+1]
        try:
            raw = G.get_edge_data(u, v)
            # lấy edge record đúng (multi-edge)
            if isinstance(raw, dict):
                # take first dict-like value
                rec = next((v for v in raw.values() if isinstance(v, dict)), None)
                data = rec if rec is not None else list(raw.values())[0]
            else:
                data = raw

            if not data:
                # fallback: thêm tọa độ node v
                nodev = G.nodes[v]
                detailed_coords.append([nodev['x'], nodev['y']])
                continue

            if 'geometry' in data:
                coords = list(data['geometry'].coords)
                # ensure coords direction: find which end is closer to u's coord
                u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
                start_pt = coords[0]; end_pt = coords[-1]
                d_start = (start_pt[0]-u_x)**2 + (start_pt[1]-u_y)**2
                d_end = (end_pt[0]-u_x)**2 + (end_pt[1]-u_y)**2
                if d_end < d_start:
                    coords = list(reversed(coords))
                # append all intermediate points (skip duplicate of u)
                for c in coords[1:]:
                    detailed_coords.append([c[0], c[1]])
            else:
                nodev = G.nodes[v]
                detailed_coords.append([nodev['x'], nodev['y']])
        except Exception:
            # bất kỳ lỗi nào -> fallback append node v
            try:
                nodev = G.nodes[v]
                detailed_coords.append([nodev['x'], nodev['y']])
            except:
                continue

    return detailed_coords
