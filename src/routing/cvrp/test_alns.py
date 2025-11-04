import sys
import os
import re
from datetime import timedelta
from numpy.random import RandomState
import numpy as np
import math # Đảm bảo đã import math

# --- SETUP ĐƯỜNG DẪN MODULE ---
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- IMPORT ---
try:
    from routing.cvrp.alns_cvrp import cvrp_helper_functions
    from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv
    from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    from routing.cvrp.alns_cvrp.destroy_operators import random_removal, worst_removal, shaw_removal
    from routing.cvrp.alns_cvrp.repair_operators import (
        best_insertion, regret_insertion,
        _calculate_route_schedule_and_feasibility,
        find_truck_by_id,
        _get_farm_info
    )
    print("✅ Import thành công!")
    from routing.cvrp.alns_cvrp.local_search_operators import apply_2_opt
except ImportError as e:
    print(f"❌ Vẫn bị lỗi Import: {e}")
    sys.exit()

# --- CẤU HÌNH ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
#INSTANCE_FILE = os.path.join(base_path, 'output_data', 'Small_structured_sample.pkl')
INSTANCE_FILE = os.path.join(base_path, 'output_data', 'haiz.pkl')
SEED, ITER = 1234, 1000
UNSERVED_PENALTY = 10000
print(f"📂 Đang đọc instance từ: {INSTANCE_FILE}")

start_temperature = 1000
end_temperature = 1
cooling_rate = 0.999

# --- 1. ĐỌC DỮ LIỆU ---
(nb_customers, capacity, dist_matrix, dist_depots, demands,
 cus_st, cus_tw, depot_tw, problem) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)

rand = RandomState(SEED)

# --- 2. TẠO LỜI GIẢI BAN ĐẦU ---
initial_schedule = compute_initial_solution(problem, rand)

# --- 3. TẠO MÔI TRƯỜNG VÀ IN KẾT QUẢ BAN ĐẦU ---
env = cvrpEnv(initial_schedule=initial_schedule, problem_instance=problem, seed=SEED)
best_solution, current_solution = env, env
best_obj = best_solution.objective()[0]
print(f"Initial Objective: {best_obj:.2f}")

destroy_operators = [random_removal, worst_removal, shaw_removal]
repair_operators = [best_insertion, regret_insertion]
random_state = np.random.RandomState(seed=SEED)

# ==============================================================================
# HÀM MÔ PHỎNG VÀ CÁC HÀM HỖ TRỢ
# ==============================================================================

def simulate_route_and_get_timeline(problem_instance, depot_idx, customer_list, shift, truck_info):
    """Mô phỏng tuyến và trả về timeline chi tiết (theo phút)."""
    if not customer_list:
        return 0, [], 0

    finish_time, is_feasible, _, _, opt_start = _calculate_route_schedule_and_feasibility(
        depot_idx, customer_list, shift, 0, problem_instance, truck_info
    )
    if not is_feasible:
        return 0, [], 0

    velocity = 1.0 if truck_info['type'] in ["Single", "Truck and Dog"] else 0.5
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    timeline = []
    current_time = opt_start
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
        timeline.append({'fid': fid, 'arrival': arrival, 'wait': wait, 'start': start_service, 'finish': finish_service})
        current_time = finish_service
        prev_idx = idx

    travel_back = depot_farm_dist[depot_idx, idx]
    travel_time_back = travel_back / velocity
    return_depot_time = current_time + travel_time_back
    return opt_start, timeline, return_depot_time

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
def print_schedule(sol):
    """In ra lịch trình tối ưu, làm tròn thời gian lên phút gần nhất."""
    prob = sol.problem_instance
    print("\n===== 🧭 LỊCH TRÌNH TỐI ƯU =====")

    for day, routes in sorted(sol.schedule.items()):
        print(f"\n📅 Ngày {day}:")

        for depot_idx, truck_id, custs, shift in routes:
            if not custs or shift == 'INTER-FACTORY':
                continue

            truck_info = find_truck_by_id(truck_id, prob['fleet']['available_trucks'])
            if not truck_info:
                continue
            
            opt_start, timeline, return_depot_time = simulate_route_and_get_timeline(
                prob, depot_idx, custs, shift, truck_info
            )

            if not timeline:
                continue

            print(f"  🚚 Truck {truck_id} ({shift}) - Depot {depot_idx}:")
            print(f"    -> Rời depot lúc {fmt(opt_start)}")

            for stop in timeline:
                # [MỚI] Làm tròn thời gian chờ LÊN phút gần nhất
                wait_min = math.ceil(stop['wait'])
                print(
                    f"    🧭 Farm {stop['fid']}: "
                    f"Arrive {fmt(stop['arrival'])}, "
                    f"Wait {wait_min} min, "
                    f"Start {fmt(stop['start'])}, "
                    f"Finish {fmt(stop['finish'])}"
                )
            
            print(f"    <- Quay về depot lúc {fmt(return_depot_time)}")

# --- 4. CHẠY ALNS ---
print("\n--- BẮT ĐẦU VÒNG LẶP ALNS ---")
temperature = start_temperature
for i in range(ITER):
    destroy_op = random_state.choice(destroy_operators)
    repair_op = random_state.choice(repair_operators)

    destroyed, unvisited = destroy_op(current_solution, random_state)

    if not unvisited: continue

    repaired, failed_to_insert = repair_op(destroyed, rand, unvisited_customers=unvisited)

    if not failed_to_insert:
        refined_solution = apply_2_opt(repaired)

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