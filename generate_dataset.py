# generate_dataset_pro.py
import osmnx as ox
import networkx as nx
import random
import pickle
import os
import time
from multiprocessing import Pool, cpu_count
from main import get_graph_data, find_best_loop_route, convert_path_to_coords

# --- CẤU HÌNH ---
TARGET_SAMPLES = 100  # Mục tiêu: 10.000 mẫu cho GNN thông minh
DATA_DIR = "data/gnn_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# Danh sách tọa độ đa dạng (Đà Nẵng, HN, HCM, Huế, Cần Thơ...)
# Để AI học được nhiều kiểu quy hoạch đô thị khác nhau
LOCATIONS = [
    # Đà Nẵng
    (16.074, 108.149),  # Đà Nẵng (Bách Khoa)
    (16.061, 108.220),  # Đà Nẵng (Cầu Rồng)
    (16.054, 108.202),  # Đà Nẵng (Bãi biển Mỹ Khê)
    (16.066, 108.224),  # Đà Nẵng (Asia Park)
    (16.078, 108.215),  # Đà Nẵng (Ngũ Hành Sơn)
    (16.068, 108.187),  # Đà Nẵng (Chợ Hàn)
    (16.059, 108.207),  # Đà Nẵng (Sông Hàn)
    (16.072, 108.189),  # Đà Nẵng (Công viên Biển Đông)

    # Huế
    (16.467, 107.590),  # Huế (Đại Nội)
    (16.458, 107.605),  # Huế (Chùa Thiên Mụ)

    # Hội An
    (15.880, 108.338),  # Hội An (Phố cổ)
    (15.875, 108.345),  # Hội An (Cầu Nhật Bản)

    # Quảng Nam
    (15.565, 108.473),  # Tam Kỳ
    (15.823, 108.320),  # Mỹ Sơn

    # Hà Nội
    (21.028, 105.854),  # Hồ Hoàn Kiếm
    (21.033, 105.850),  # Văn Miếu

    # TP. Hồ Chí Minh
    (10.776, 106.700),  # Dinh Độc Lập
    (10.762, 106.682),  # Chợ Bến Thành

    # Nha Trang
    (12.238, 109.196),  # Bãi biển Nha Trang
    (12.238, 109.191),  # Vinpearl Nha Trang

    # Phú Quốc
    (10.226, 103.963),  # Dương Đông
    (10.210, 103.975),  # Bãi Sao
]


def generate_one_sample(_):
    """Hàm sinh 1 mẫu dữ liệu (được chạy song song)"""
    try:
        # 1. Random vị trí và bán kính
        base_lat, base_lng = random.choice(LOCATIONS)
        # Randomize vị trí ±1.5km để không bị trùng lặp map
        lat = base_lat + random.uniform(-0.015, 0.015)
        lng = base_lng + random.uniform(-0.015, 0.015)
        
        # Random mục tiêu: 3km, 5km, 7km, 10km
        target_km = random.choice([3.0, 4.0, 5.0, 7.0, 10.0])
        target_m = target_km * 1000

        # 2. Tải Graph (Silent mode để đỡ rác log)
        # dist 2000m là đủ cho 10km loop
        G = ox.graph_from_point((lat, lng), dist=2000, network_type='walk', simplify=True)
        G_undirected = G.to_undirected()

        if len(G_undirected.nodes) < 100: return None # Bỏ qua graph quá nhỏ

        # 3. Chọn điểm bắt đầu ngẫu nhiên
        start_node = random.choice(list(G_undirected.nodes))

        # 4. Chạy thuật toán "Smart Walk" (Làm Label)
        # Tăng iterations lên 50 để đảm bảo label chất lượng cao
        best_path, best_len = find_best_loop_route(
            G_undirected, start_node, target_m, 
            num_iterations=50, 
            max_nodes_per_walk=80
        )

        # 5. Kiểm tra chất lượng mẫu
        # Chỉ lấy mẫu nếu sai số độ dài < 15% và là vòng lặp
        if best_path and abs(best_len - target_m) < target_m * 0.15:
            # Kiểm tra vòng lặp (đầu == cuối)
            if best_path[0] == best_path[-1]:
                
                # Lưu dữ liệu thô để sau này biến đổi thành Tensor cho GNN
                return {
                    "graph_nodes": {n: (G_undirected.nodes[n]['x'], G_undirected.nodes[n]['y']) for n in G_undirected.nodes},
                    "graph_edges": list(G_undirected.edges(data='length')),
                    "start_node": start_node,
                    "target_distance": target_m,
                    "label_path": best_path, # Đây là output mà GNN cần học để dự đoán
                    "actual_distance": best_len
                }
        
        return None # Không tìm được đường tốt
    except Exception:
        return None

def main():
    start_time = time.time()
    print(f"🚀 Bắt đầu sinh {TARGET_SAMPLES} mẫu dữ liệu...")
    print(f"💻 Sử dụng {cpu_count()} nhân CPU để chạy song song.")

    valid_samples = []
    
    # Sử dụng Pool để chạy đa luồng
    with Pool(processes=cpu_count()) as pool:
        # Thử chạy 1.5 lần mục tiêu vì sẽ có mẫu bị lỗi/bỏ qua
        results = pool.imap_unordered(generate_one_sample, range(int(TARGET_SAMPLES * 1.5)))
        
        for i, sample in enumerate(results):
            if sample:
                valid_samples.append(sample)
                if len(valid_samples) % 100 == 0:
                    print(f"✅ Đã tạo được: {len(valid_samples)}/{TARGET_SAMPLES} mẫu ({(time.time() - start_time)/60:.1f} phút)")
                    
                    # Lưu checkpoint để lỡ tắt máy không mất hết
                    with open(f"{DATA_DIR}/dataset_checkpoint_{len(valid_samples)}.pkl", "wb") as f:
                        pickle.dump(valid_samples, f)
            
            if len(valid_samples) >= TARGET_SAMPLES:
                break

    # Lưu file cuối cùng
    with open(f"{DATA_DIR}/final_dataset_10k.pkl", "wb") as f:
        pickle.dump(valid_samples, f)
        
    total_time = (time.time() - start_time) / 3600
    print(f"🎉 HOÀN THÀNH! {len(valid_samples)} mẫu trong {total_time:.2f} giờ.")

if __name__ == "__main__":
    # Windows cần dòng này để chạy multiprocessing
    main()