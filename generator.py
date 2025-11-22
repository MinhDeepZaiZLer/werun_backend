import osmnx as ox
import networkx as nx
import random
import pickle
import os
import time
import gc # Garbage Collector
from multiprocessing import Pool
from tqdm import tqdm
from math import sqrt

# --- CẤU HÌNH AN TOÀN CHO RAM ---
TARGET_SAMPLES = 10000
DATA_DIR = "./gnn_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# Danh sách địa điểm
LOCATIONS = [
    (16.061, 108.220), # Đà Nẵng
    (21.028, 105.854), # Hà Nội
    (10.776, 106.700), # HCM
]
def get_graph_data(lat, lng, dist=2000):
    # simplify=True cực quan trọng để giảm RAM
    G = ox.graph_from_point((lat, lng), dist=dist, network_type='walk', simplify=True)
    G_connected = ox.truncate.largest_component(G, strongly=True)
    return G_connected.to_undirected()

def calculate_path_length(G, path):
    total = 0
    for i in range(len(path) - 1):
        try:
            total += G.get_edge_data(path[i], path[i+1])[0].get('length', 0)
        except: continue
    return total

def create_smart_walk(G, start_node, target_length, max_nodes=60):
    path = [start_node]; curr = start_node; curr_len = 0
    for _ in range(max_nodes):
        if curr_len >= target_length: break
        neighbors = list(G.neighbors(curr))
        if not neighbors: break
        
        candidates = []; weights = []
        for n in neighbors:
            score = 1.0
            if len(path) > 1 and n == path[-2]: score = 0.01
            elif G.degree(n) == 1: score = 0.1
            elif n in path: score = 0.2
            candidates.append(n); weights.append(score)
            
        if sum(weights) == 0: weights = [1.0] * len(candidates)
        next_node = random.choices(candidates, weights=weights, k=1)[0]
        
        try:
            edge_len = G.get_edge_data(curr, next_node)[0].get('length', 0)
            curr_len += edge_len
            path.append(next_node); curr = next_node
        except: break
    return path

def find_best_loop_route(G, start_node, target_m, num_iterations=30):
    best_path = []; best_fit = -1; best_len = 0
    walk_dist = target_m * 0.55
    
    for i in range(num_iterations):
        path_out = create_smart_walk(G, start_node, walk_dist, max_nodes=80)
        if len(path_out) < 3: continue
        last = path_out[-1]
        try:
            path_ret = nx.shortest_path(G, last, start_node, weight='length')
            full = path_out + path_ret[1:]
            length = calculate_path_length(G, full)
            diff = abs(length - target_m)
            fit = 1.0 / (diff + 1.0)
            if fit > best_fit:
                best_fit = fit; best_path = full; best_len = length
        except: continue
        
    if best_path and abs(best_len - target_m) < target_m * 0.2:
        return best_path, best_len
    
    # Fallback Out-and-Back
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
    worker_G = None 

def worker_init():
    """Hàm này chạy 1 lần khi Worker khởi động"""
    global worker_G
    try:
        # Mỗi worker chọn ngẫu nhiên 1 thành phố để "chuyên trách"
        lat, lng = random.choice(LOCATIONS)
        lat += random.uniform(-0.01, 0.01)
        lng += random.uniform(-0.01, 0.01)
        worker_G = get_graph_data(lat, lng, dist=2500)
    except:
        worker_G = None

def generate_one_sample_safe(_):
    """Hàm sinh mẫu dùng map có sẵn trong RAM của Worker"""
    global worker_G
    if worker_G is None: return None
    
    try:
        target_m = random.choice([3000, 5000, 7000, 10000])
        start_node = random.choice(list(worker_G.nodes))
        
        best_path, best_len = find_best_loop_route(worker_G, start_node, target_m, num_iterations=40)

        if best_path and abs(best_len - target_m) < target_m * 0.15:
            if best_path[0] == best_path[-1]:
                return {
                    "graph_nodes": {n: (worker_G.nodes[n]['x'], worker_G.nodes[n]['y']) for n in worker_G.nodes},
                    "graph_edges": list(worker_G.edges(data='length')),
                    "start_node": start_node,
                    "target_distance": target_m,
                    "label_path": best_path,
                    "actual_distance": best_len
                }
        return None
    except: return None
def main():
    start_time = time.time()
    
    # Kaggle có 2 vCPU, dùng 2 là tối ưu nhất
    NUM_WORKERS = 2 
    print(f"🚀 Bắt đầu sinh {TARGET_SAMPLES} mẫu (Safe Mode - {NUM_WORKERS} Workers)...")

    # Chia nhỏ thành các batch để xả RAM
    BATCH_SIZE = 1000
    num_batches = TARGET_SAMPLES // BATCH_SIZE
    
    total_collected = 0

    for batch_idx in range(num_batches):
        print(f"\n📦 Đang xử lý Batch {batch_idx + 1}/{num_batches}...")
        batch_samples = []
        
        # Khởi tạo Pool mới cho mỗi batch để reset bộ nhớ
        with Pool(processes=NUM_WORKERS, initializer=worker_init) as pool:
            # Chạy dư 50% để bù trừ
            results = pool.imap_unordered(generate_one_sample_safe, range(int(BATCH_SIZE * 1.5)))
            
            with tqdm(total=BATCH_SIZE) as pbar:
                for sample in results:
                    if sample:
                        batch_samples.append(sample)
                        pbar.update(1)
                    
                    if len(batch_samples) >= BATCH_SIZE:
                        pool.terminate()
                        break
        
        # Lưu batch ra file riêng
        filename = f"{DATA_DIR}/part_{batch_idx}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(batch_samples, f)
        
        print(f"💾 Đã lưu {len(batch_samples)} mẫu vào {filename}")
        total_collected += len(batch_samples)
        
        # Xả bộ nhớ
        del batch_samples
        gc.collect() 
    
    print(f"\n🎉 HOÀN THÀNH TẤT CẢ! Tổng: {total_collected} mẫu.")
    print(f"Dữ liệu nằm trong thư mục: {DATA_DIR}")
    print("💡 Mẹo: Hãy dùng shutil.make_archive để nén folder này lại và tải về.")

if __name__ == "__main__":
    main()