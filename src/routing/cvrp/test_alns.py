import sys
import os
import re
import math
from datetime import timedelta
from numpy.random import RandomState
import numpy as np

# --- SETUP ĐƯỜNG DẪN MODULE ---
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- IMPORT ---
try:
    from routing.cvrp.alns_cvrp import cvrp_helper_functions
    from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
    from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    from routing.cvrp.alns_cvrp.destroy_operators import random_removal, worst_removal, shaw_removal, time_worst_removal
    from routing.cvrp.alns_cvrp.repair_operators import best_insertion, regret_insertion, time_shift_repair
    from routing.cvrp.alns_cvrp.local_search_operators import apply_2_opt, apply_relocate, apply_exchange
    # Import các hàm tiện ích cần thiết
    from routing.cvrp.alns_cvrp.utils import _calculate_route_schedule_and_feasibility, _get_farm_info, find_truck_by_id
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Vẫn bị lỗi Import: {e}")
    sys.exit()

# --- CẤU HÌNH ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
#INSTANCE_FILE = os.path.join(base_path, 'output_data', 'haiz.pkl')
INSTANCE_FILE = os.path.join(base_path, 'output_data', 'Small_structured_sample.pkl')
#INSTANCE_FILE = os.path.join(base_path, 'output_data', 'CEL_structured_instance.pkl')
SEED, ITER = 1234, 1000

# CẤU HÌNH SIMULATED ANNEALING
start_temperature = 100
end_temperature = 0.1
cooling_rate = 0.999

print(f"📂 Đang đọc instance từ: {INSTANCE_FILE}")

# --- 1. ĐỌC DỮ LIỆU ---
(nb_customers, capacity, dist_matrix, dist_depots, demands,
 cus_st, cus_tw, depot_tw, problem) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)

rand = RandomState(SEED)

# --- 2. TẠO LỜI GIẢI BAN ĐẦU (Đã đơn giản hóa) ---
initial_schedule = compute_initial_solution(problem, rand)

# --- 3. TẠO MÔI TRƯỜNG ---
env = cvrpEnv(initial_schedule=initial_schedule, problem_instance=problem, seed=SEED)
best_solution, current_solution = env, env
best_obj = best_solution.objective()[0]
print(f"Initial Objective: {best_obj:.2f}")

destroy_operators = [random_removal, worst_removal, shaw_removal, time_worst_removal]
repair_operators = [best_insertion, regret_insertion, time_shift_repair]
random_state = np.random.RandomState(seed=SEED)
# ==============================================================================
# HÀM MÔ PHỎNG VÀ CÁC HÀM HỖ TRỢ
# ==============================================================================
def apply_full_local_search(solution):
    """Áp dụng một chuỗi các toán tử Local Search."""
    # Chạy các toán tử nội tuyến trước
    solution = apply_relocate(solution)
    solution = apply_2_opt(solution)
    # Chạy toán tử liên tuyến để gộp/tái cấu trúc
    solution = apply_exchange(solution)
    return solution


def simulate_route_and_get_timeline(problem_instance, depot_idx, customer_list, shift, truck_info):
    """Mô phỏng tuyến thực tế đúng với logic objective (không delay start)."""
    if not customer_list:
        return 0, [], 0

    # Giống hệt logic trong objective: start_time_at_depot = 0
    start_time_at_depot = 0
    finish_time, is_feasible, total_dist, total_wait, opt_start = _calculate_route_schedule_and_feasibility(
        depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info
    )
    if not is_feasible:
        return 0, [], 0

    # Lấy thông tin để in (arrival, start, finish, wait) theo đúng dòng tính của hàm đó
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    velocity = 1.0 if truck_info['type'] in ["Single", "Truck and Dog"] else 0.5

    timeline = []
    current_time = start_time_at_depot
    prev_idx = -1

    for i, fid in enumerate(customer_list):
        idx, details, demand = _get_farm_info(fid, problem_instance)
        travel_dist = depot_farm_dist[depot_idx, idx] if i == 0 else dist_matrix[prev_idx, idx]
        travel_time = travel_dist / velocity
        arrival = current_time + travel_time
        start_tw, _ = details['time_windows'][shift]
        wait = max(0, start_tw - arrival)
        start_service = arrival + wait
        fix, var = details['service_time_params']
        service_duration = fix + (demand / var if var > 0 else 0)
        finish_service = start_service + service_duration
        timeline.append({
            'fid': fid,
            'arrival': arrival,
            'wait': wait,
            'start': start_service,
            'finish': finish_service
        })
        current_time = finish_service
        prev_idx = idx

    # Quay về depot
    travel_back = depot_farm_dist[depot_idx, prev_idx]
    travel_time_back = travel_back / velocity
    return_depot_time = current_time + travel_time_back

    return start_time_at_depot, timeline, return_depot_time


def _clean_base_id(fid):
    """Chuẩn hóa farm_id, tách phần gốc."""
    if isinstance(fid, (int, float)):
        return str(int(fid))
    return re.split(r'(_onfly.*|_part.*|_d\d+)', str(fid))[0]

def find_truck_by_id(truck_id, available_trucks):
    """Trả về thông tin truck theo ID."""
    for t in available_trucks:
        if t['id'] == truck_id:
            return t
    return None

# <<< HÀM fmt ĐÃ ĐƯỢC CẬP NHẬT ĐỂ LÀM TRÒN LÊN PHÚT >>>
def fmt(minutes):
    """Định dạng phút (float) sang chuỗi HH:MM, làm tròn LÊN phút gần nhất."""
    if minutes is None or not isinstance(minutes, (int, float)):
        return "00:00"
    
    # Làm tròn TỔNG SỐ PHÚT lên số nguyên gần nhất
    total_rounded_minutes = math.ceil(minutes)
    
    # Tính toán giờ và phút từ tổng số phút đã làm tròn
    hours, mins = divmod(total_rounded_minutes, 60)
    
    # Định dạng chuỗi đầu ra
    return f"{int(hours):02d}:{int(mins):02d}"

# <<< HÀM IN KHÔNG THAY ĐỔI CẤU TRÚC, CHỈ THAY ĐỔI CÁCH LÀM TRÒN >>>
# (Dán vào file test_alns.py, thay thế hàm print_schedule cũ)

def print_schedule(sol):
    """
    ## SIMPLIFIED & CORRECTED for 5-element tuple ##
    In ra lịch trình tối ưu cho một ngày.
    (Đã cập nhật để in chi tiết INTER-FACTORY)
    """
    prob = sol.problem_instance
    print("\n===== 🧭 LỊCH TRÌNH TỐI ƯU CHO NGÀY =====")
    
    # Lấy ma trận khoảng cách depot (cần cho INTER-FACTORY)
    depot_dist_matrix = prob.get('distance_matrix_facilities')

    # <<< SỬA Ở ĐÂY: Đổi tên `_` thành `start_time` để sử dụng >>>
    for depot, truck_id, custs, shift, start_time in sol.schedule:
        if not custs: continue # Bỏ qua nếu tuyến rỗng

        truck_info = find_truck_by_id(truck_id, prob['fleet']['available_trucks'])
        if not truck_info:
             print(f"   ⚠️ Lỗi: Không tìm thấy Truck {truck_id}")
             continue

        if shift == 'INTER-FACTORY':
            # <<< LOGIC MỚI ĐỂ IN CHI TIẾT INTER-FACTORY >>>
            try:
                # 1. Parse custs string
                parts = str(custs[0]).split('_')
                source_depot_idx = int(parts[2])
                target_depot_idx = int(parts[4])

                if depot_dist_matrix is None:
                    raise ValueError("Thiếu 'distance_matrix_facilities' trong problem_instance")
                
                # 2. Tính toán thời gian
                travel_dist = depot_dist_matrix[source_depot_idx][target_depot_idx]
                truck_name = truck_info['type']
                velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5
                travel_time = travel_dist / velocity if velocity > 0 else float('inf')
                
                arrival_time = start_time + travel_time

                # 3. In ra
                print(f"   🏭 Truck {truck_id} ({shift}) - Từ Depot {source_depot_idx} đến Depot {target_depot_idx}:")
                # Dùng fmt() để làm tròn thời gian
                print(f"       - Xuất phát (Depot {source_depot_idx}): {fmt(start_time)}")
                print(f"       - Đến nơi (Depot {target_depot_idx}): {fmt(arrival_time)} (Di chuyển {fmt(travel_time)})")

            except Exception as e:
                # In dự phòng nếu parse lỗi hoặc thiếu ma trận
                print(f"   🏭 Truck {truck_id} ({shift}): {str(custs[0]).replace('_', ' ')} (Lỗi tính toán: {e})")
            
            # (Đã bỏ 'continue' ở đây)

        else:
            # <<< LOGIC CŨ CHO FARM (CÓ ĐIỀU CHỈNH) >>>
            
            # Gọi hàm mô phỏng (hàm này đang giả định start=0)
            optimal_start_calc, timeline, return_depot_time = simulate_route_and_get_timeline(prob, depot, custs, shift, truck_info)
            
            if not timeline: continue

            # In thời gian xuất phát THỰC TẾ (lấy từ tuple, không phải từ hàm simulate)
            print(f"   🚚 Truck {truck_id} ({shift}) - Depot {depot} (Xuất phát lúc {fmt(start_time)}):")
            
            # Dịch chuyển (offset) timeline dựa trên thời gian xuất phát thực tế
            for stop in timeline:
                # optimal_start_calc là 0 (do hardcode trong hàm simulate)
                # Ta cộng chênh lệch (start_time) vào timeline
                offset_arrival = start_time + (stop['arrival'] - optimal_start_calc)
                offset_start = start_time + (stop['start'] - optimal_start_calc)
                offset_finish = start_time + (stop['finish'] - optimal_start_calc)

                print(f"       🧭 Farm {stop['fid']}: Arrive {fmt(offset_arrival)}, Wait {stop['wait']:.0f} min, "
                      f"Start {fmt(offset_start)}, Finish {fmt(offset_finish)}")

# --- 4. CHẠY ALNS (Đã đơn giản hóa) ---
print("\n--- BẮT ĐẦU VÒNG LẶP ALNS ---")
temperature = start_temperature

for i in range(ITER):
    destroy_op = random_state.choice(destroy_operators)
    repair_op = random_state.choice(repair_operators)
    
    # ## SIMPLIFICATION: unvisited bây giờ là danh sách các farm_id
    destroyed, unvisited = destroy_op(current_solution, random_state)
    
    if not unvisited: continue
    
    # Lọc ra các ID 'TRANSFER_' nếu có (dù không nên có)
    farms_to_reinsert = [c for c in unvisited if not str(c).startswith('TRANSFER_')]
    if not farms_to_reinsert: continue
        
    repaired, failed_to_insert = repair_op(destroyed, rand, unvisited_customers=farms_to_reinsert)
    
    if not failed_to_insert:
        
        refined_solution = apply_full_local_search(repaired)

        current_obj = current_solution.objective()[0]
        refined_obj = refined_solution.objective()[0]

        if refined_obj < best_obj:
            best_solution = refined_solution
            best_obj = refined_obj
            current_solution = refined_solution
            print(f"Iter {i}: New best found! Obj = {best_obj:.2f}")
        
        elif random_state.random() < math.exp((current_obj - refined_obj) / temperature):
             current_solution = refined_solution

    temperature = max(end_temperature, temperature * cooling_rate)

print(f"\n🏁 Final Best Objective: {best_solution.objective()[0]:.2f}")

# --- 5. IN LỊCH TRÌNH TỐI ƯU CUỐI CÙNG ---
print_schedule(best_solution)