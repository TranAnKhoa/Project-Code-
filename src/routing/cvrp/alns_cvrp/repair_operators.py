import copy
import random
import numpy as np
import re
from collections import defaultdict
import itertools
# ==============================================================================
# HÀM TIỆN ÍCH - CỐT LÕI CỦA VIỆC SỬA LỖI
# ==============================================================================

def _clean_base_id(fid):
    """Làm sạch ID để lấy ID gốc của nông trại vật lý."""
    if not isinstance(fid, str):
        return fid
    return re.split(r'(_onfly.*|_fallback_part.*|_part.*|_d\d+)', fid)[0]

def _get_farm_info(farm_id, problem_instance):
    """Hàm "thông dịch" ID, trả về thông tin chính xác cho cả farm thật và ảo."""
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map']
    virtual_map = problem_instance.get('virtual_split_farms', {})
    farms = problem_instance['farms']
    
    base_id = _clean_base_id(farm_id)
    
    try:
        farm_idx = farm_id_to_idx_map[base_id]
    except KeyError:
        try:
            farm_idx = farm_id_to_idx_map[int(base_id)]
        except (KeyError, ValueError):
            raise KeyError(f"Không thể tìm thấy Farm ID '{base_id}' (từ '{farm_id}') trong map.")
            
    farm_details = farms[farm_idx]
    
    if farm_id in virtual_map:
        demand = virtual_map[farm_id]['portion']
    else:
        demand = farm_details['demand']
        
    return farm_idx, farm_details, demand

# --- HÀM TÌM XE (Giữ nguyên) ---
def find_truck_by_id(truck_id, available_trucks):
    """Tiện ích để tìm thông tin chi tiết của xe từ ID."""
    for truck in available_trucks:
        if truck['id'] == truck_id:
            return truck
    return None

# --- CÁC HÀM TÍNH TOÁN ĐÃ ĐƯỢC SỬA LỖI ---

def _calculate_route_schedule_and_feasibility(depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info):
    """
    Tính toán lịch trình, kiểm tra feasibility VÀ trả về tổng quãng đường, tổng thời gian chờ, thời gian xuất phát tối ưu.
    """
    if not customer_list:
        return start_time_at_depot, True, 0, 0, start_time_at_depot

    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist = problem_instance['distance_depots_farms']
    farms = problem_instance['farms']
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']
    depot_end_time = 1440
    truck_name = truck_info['type']
    velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5
    virtual_map = problem_instance.get('virtual_split_farms', {})

    def _resolve_farm(fid):
        base_id_str = _clean_base_id(fid)
        try: base_idx = farm_id_to_idx[base_id_str]
        except KeyError: base_idx = farm_id_to_idx[int(base_id_str)]
        base_info = farms[base_idx]
        if isinstance(fid, str) and fid in virtual_map:
            return base_idx, virtual_map[fid]['portion'], base_info['service_time_params'], base_info['time_windows']
        else:
            return base_idx, base_info['demand'], base_info['service_time_params'], base_info['time_windows']

    timeline_sim = []
    current_time_sim1 = start_time_at_depot
        # --- 🔧 Điều chỉnh thời gian khởi hành để tránh chờ lâu ---
    if customer_list:
        first_farm_id = customer_list[0]
        base_id_str = _clean_base_id(first_farm_id)
        try:
            first_farm_idx = farm_id_to_idx[base_id_str]
        except KeyError:
            first_farm_idx = farm_id_to_idx[int(base_id_str)]
        first_tw_start, _ = farms[first_farm_idx]['time_windows'][shift]
        travel_from_depot = depot_farm_dist[depot_idx, first_farm_idx] / velocity
        # Cập nhật thời gian xuất phát hợp lý
        start_time_at_depot = max(start_time_at_depot, first_tw_start - travel_from_depot)
        current_time_sim1 = start_time_at_depot

    idx, demand, params, tw = _resolve_farm(customer_list[0])
    travel_time = depot_farm_dist[depot_idx, idx] / velocity
    arrival = current_time_sim1 + travel_time
    start_tw, end_tw = tw[shift]
    if arrival > end_tw: return -1, False, -1, -1, -1
    service_start = max(arrival, start_tw)
    service_duration = params[0] + (demand / params[1] if params[1] > 0 else 0)
    current_time_sim1 = service_start + service_duration
    timeline_sim.append({'arrival': arrival, 'start': service_start})

    for i in range(len(customer_list) - 1):
        from_idx, _, _, _ = _resolve_farm(customer_list[i])
        to_idx, to_demand, to_params, to_tw = _resolve_farm(customer_list[i+1])
        travel_time = dist_matrix[from_idx, to_idx] / velocity
        arrival = current_time_sim1 + travel_time
        start_tw, end_tw = to_tw[shift]
        if arrival > end_tw: return -1, False, -1, -1, -1
        service_start = max(arrival, start_tw)
        service_duration = to_params[0] + (to_demand / to_params[1] if to_params[1] > 0 else 0)
        current_time_sim1 = service_start + service_duration
        timeline_sim.append({'arrival': arrival, 'start': service_start})
    
    last_idx, _, _, _ = _resolve_farm(customer_list[-1])
    travel_time_back = depot_farm_dist[depot_idx, last_idx] / velocity
    finish_time_sim1 = current_time_sim1 + travel_time_back
    if finish_time_sim1 > depot_end_time: return -1, False, -1, -1, -1

    slacks = [t['start'] - t['arrival'] for t in timeline_sim]
    min_slack = min(slacks) if slacks else 0
    optimal_start_time = start_time_at_depot + min_slack

    total_dist = 0; total_wait = 0
    current_time_final = optimal_start_time

    idx, demand, params, tw = _resolve_farm(customer_list[0])
    travel_dist = depot_farm_dist[depot_idx, idx]; total_dist += travel_dist
    travel_time = travel_dist / velocity; arrival = current_time_final + travel_time
    start_tw, _ = tw[shift]; wait_time = max(0, start_tw - arrival); total_wait += wait_time
    service_start = arrival + wait_time
    service_duration = params[0] + (demand / params[1] if params[1] > 0 else 0)
    current_time_final = service_start + service_duration

    for i in range(len(customer_list) - 1):
        from_idx, _, _, _ = _resolve_farm(customer_list[i])
        to_idx, to_demand, to_params, to_tw = _resolve_farm(customer_list[i+1])
        travel_dist = dist_matrix[from_idx, to_idx]; total_dist += travel_dist
        travel_time = travel_dist / velocity; arrival = current_time_final + travel_time
        start_tw, _ = to_tw[shift]; wait_time = max(0, start_tw - arrival); total_wait += wait_time
        service_start = arrival + wait_time
        service_duration = to_params[0] + (to_demand / to_params[1] if to_params[1] > 0 else 0)
        current_time_final = service_start + service_duration
    
    last_idx, _, _, _ = _resolve_farm(customer_list[-1])
    travel_dist_back = depot_farm_dist[depot_idx, last_idx]; total_dist += travel_dist_back
    travel_time_back = travel_dist_back / velocity
    finish_time_final = current_time_final + travel_time_back
        
    return finish_time_final, True, total_dist, total_wait, optimal_start_time

def calculate_insertion_impact(depot_idx, customer_list, farm_id_to_insert, insert_pos, shift, problem_instance, start_time=0):
    """
    Tính toán chi phí tăng thêm (distance + waiting) khi chèn farm_id_to_insert vào một tuyến cụ thể.
    Trả về (Δcost, is_feasible).
    """
    WAIT_COST_PER_MIN = problem_instance.get('waiting_cost_per_min', 0.2)

    # --- cost parameters ---
    facilities = problem_instance['facilities']
    farms = problem_instance['farms']
    farm_map = problem_instance['farm_id_to_idx_map']
    dist = problem_instance['distance_matrix_farms']
    dep_farm = problem_instance['distance_depots_farms']
    costs = problem_instance['costs']

    # Tạo tuyến mới sau khi chèn
    new_route = customer_list[:insert_pos] + [farm_id_to_insert] + customer_list[insert_pos:]

    # --- Gọi hàm feasibility trước & sau khi chèn ---
    base_stats = _calculate_route_schedule_and_feasibility(depot_idx, customer_list, shift, start_time, problem_instance)
    new_stats = _calculate_route_schedule_and_feasibility(depot_idx, new_route, shift, start_time, problem_instance)

    if base_stats is None or new_stats is None:
        return float('inf'), False

    _, base_feas, base_dist, base_wait, _ = base_stats
    _, new_feas, new_dist, new_wait, _ = new_stats

    if not new_feas:
        return float('inf'), False

    # --- Chi phí biến theo loại xe + vùng ---
    if customer_list:
        first_farm_id = customer_list[0]
    else:
        first_farm_id = farm_id_to_insert
    farm_idx, farm_info, _ = _get_farm_info(first_farm_id, problem_instance)

    # chọn region & type tạm (giả định depot có region)
    depot_region = facilities[depot_idx].get('region', None)
    truck_type = 'Single'
    var_cost_per_km = costs['variable_cost_per_km'].get((truck_type, depot_region), 1.0)

    # --- Tính Δcost ---
    delta_dist = new_dist - base_dist
    delta_wait = new_wait - base_wait
    delta_cost = delta_dist * var_cost_per_km + delta_wait * WAIT_COST_PER_MIN

    return delta_cost, True


# --- HÀM SIÊU TRỢ GIÚP ĐÃ ĐƯỢC SỬA LỖI ---

def _check_insertion_feasibility(problem_instance, route_info, insert_pos, farm_id_to_insert, shift, start_time=0):
    """Thực hiện The Feasibility Checklist và tính toán chi phí tăng thêm."""
    depot_idx, truck_id, customer_list, _ = route_info
    truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
    
    WAIT_COST_PER_MIN = 0.2
    var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get((truck_info['type'], truck_info['region']), 1.0)
    
    type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
    truck_type_idx = type_to_idx.get(truck_info['type'])
    if truck_type_idx is None: return False, float('inf'), -1

    _, farm_details, farm_demand = _get_farm_info(farm_id_to_insert, problem_instance)
    farm_access = farm_details.get('accessibility')
    if farm_access is None or len(farm_access) <= truck_type_idx or farm_access[truck_type_idx] != 1:
        return False, float('inf'), -1

    current_load = sum(_get_farm_info(fid, problem_instance)[2] for fid in customer_list)
    if current_load + farm_demand > truck_info['capacity']:
        return False, float('inf'), -1

    old_total_cost = 0
    if customer_list:
        _, is_feasible_old, old_dist, old_wait, _ = _calculate_route_schedule_and_feasibility(
            depot_idx, customer_list, shift, start_time, problem_instance, truck_info=truck_info
        )
        if not is_feasible_old: return False, float('inf'), -1
        old_total_cost = (old_dist * var_cost_per_km) + (old_wait * WAIT_COST_PER_MIN)

    test_route = customer_list[:insert_pos] + [farm_id_to_insert] + customer_list[insert_pos:]
    new_finish_time, is_feasible_new, new_dist, new_wait, _ = _calculate_route_schedule_and_feasibility(
        depot_idx, test_route, shift, start_time, problem_instance, truck_info=truck_info
    )

    if not is_feasible_new:
        return False, float('inf'), -1

    new_total_cost = (new_dist * var_cost_per_km) + (new_wait * WAIT_COST_PER_MIN)
    cost_increase = new_total_cost - old_total_cost
        
    return True, cost_increase, new_finish_time



def _get_all_insertions_for_farm(schedule, farm_id_to_insert, problem_instance, random_state, target_day):
    """
    Tìm tất cả vị trí chèn khả thi cho một farm, CHỈ TRONG NGÀY MỤC TIÊU (target_day).
    """
    farm_idx, farm_details, farm_demand = _get_farm_info(farm_id_to_insert, problem_instance)
    
    available_trucks = problem_instance['fleet']['available_trucks']
    truck_id_map = {truck['id']: truck for truck in available_trucks}
    possible_insertions = []
    day_idx = target_day # Chỉ xét ngày mục tiêu

    routes_of_day = schedule[day_idx]
    truck_finish_info = {} 

    # --- PHẦN 1: TÍNH TOÁN VÀ THỬ CHÈN VÀO TUYẾN CŨ TRONG NGÀY ---
    for route_idx, route_info in enumerate(routes_of_day):
        depot_idx, truck_id, customer_list, existing_shift = route_info
        
        if existing_shift == 'INTER-FACTORY': continue
        
        finish_time, _ = _calculate_route_schedule_and_feasibility(depot_idx, customer_list, existing_shift, 0, problem_instance)
        
        current_finish_time, _ = truck_finish_info.get(truck_id, (0.0, -1))
        if finish_time > current_finish_time:
            truck_finish_info[truck_id] = (finish_time, depot_idx)

            truck_details = truck_id_map.get(truck_id)
            if not truck_details: continue
            
            current_load = 0
            for c_id in customer_list:
                _, _, demand = _get_farm_info(c_id, problem_instance)
                current_load += demand

            if current_load + farm_demand > truck_details['capacity']: continue
            
            truck_type_idx = ['Single', '20m', '26m', 'Truck and Dog'].index(truck_details['type'])
            if not farm_details['accessibility'][truck_type_idx]: continue
            
            for insert_pos in range(len(customer_list) + 1):
                cost_increase, is_feasible = calculate_insertion_impact(
                    depot_idx, customer_list, farm_id_to_insert, insert_pos, existing_shift,
                    problem_instance
                )
                if is_feasible:
                    possible_insertions.append(
                        (cost_increase, day_idx, route_idx, insert_pos, existing_shift, truck_id, depot_idx)
                    )

        # --- PHẦN 2: THỬ TẠO CHUYẾN ĐI MỚI (MULTI-TRIP) ---
        trucks_used_today = set(truck_finish_info.keys())
        for truck_id in trucks_used_today:
            truck_details = truck_id_map.get(truck_id)
            if not truck_details or farm_demand > truck_details['capacity']: continue
            
            start_time_for_new_trip, depot_idx_for_new_trip = truck_finish_info[truck_id]
            start_time_for_new_trip += 30 
            
            for shift in ['AM', 'PM']:
                 _, is_feasible = _calculate_route_schedule_and_feasibility(
                     depot_idx_for_new_trip, [farm_id_to_insert], shift, start_time_for_new_trip, problem_instance
                 )
                 if is_feasible:
                     new_route_cost_info = _create_new_route_for_farm(farm_id_to_insert, problem_instance, force_depot_idx=depot_idx_for_new_trip)
                     if new_route_cost_info:
                         cost_increase, _, _, _ = new_route_cost_info
                         possible_insertions.append(
                             (cost_increase, day_idx, -1, 0, shift, truck_id, depot_idx_for_new_trip)
                         )
    return possible_insertions

def _find_best_insert_for_visit(schedule_on_day, visit_id, problem_instance):
    """
    Tìm vị trí chèn tốt nhất cho một visit (1 ID, có thể là virtual) trong một ngày.
    Trả về dict giống structure cũ: {'cost', 'route_idx', 'pos', 'shift', 'new_route_details'}
    """

    # --- config chi phí ---
    WAIT_COST_PER_MIN = problem_instance.get('waiting_cost_per_min', 0.2)
    type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}

    farms = problem_instance['farms']
    farm_map = problem_instance['farm_id_to_idx_map']
    dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm = problem_instance['distance_depots_farms']
    virtual_map = problem_instance.get('virtual_split_farms', {})

    # helper: resolve base id and index + demand + service params + time_windows
    def resolve(fid):
        # returns (idx, demand, service_time_params, time_windows)
        if isinstance(fid, str) and fid in virtual_map:
            base = virtual_map[fid]['base_id']
            portion = virtual_map[fid].get('portion', 0)
            base_clean = re.split(r'(_onfly.*|_fallback_part.*|_part.*|_d\d+)', base)[0]
            try:
                idx = farm_map[base_clean]
            except KeyError:
                idx = farm_map[int(base_clean)]
            info = farms[idx]
            return idx, portion, info['service_time_params'], info['time_windows']
        else:
            base_clean = re.split(r'(_onfly.*|_fallback_part.*|_part.*|_d\d+)', str(fid))[0]
            try:
                idx = farm_map[base_clean]
            except KeyError:
                idx = farm_map[int(base_clean)]
            info = farms[idx]
            return idx, info['demand'], info['service_time_params'], info['time_windows']

    # helper: compute route metrics given route list and truck_info
    def compute_route_metrics(depot_idx, route_list, shift, start_time_at_depot, truck_info):
        """
        Trả về (feasible:boolean, finish_time, total_travel_time, total_wait_minutes)
        travel times use depot_farm and dist_matrix and are divided by velocity like other code.
        """
        if not route_list:
            return True, start_time_at_depot, 0.0, 0.0

        vel = 1.0 if truck_info and truck_info.get('type') in ["Single", "Truck and Dog"] else 0.5
        current_time = start_time_at_depot
        total_travel = 0.0
        total_wait = 0.0

        # to first
        first = route_list[0]
        first_idx, first_demand, first_params, first_tw = resolve(first)
        travel = depot_farm[depot_idx, first_idx] / vel
        total_travel += travel
        arrival = current_time + travel
        start_tw, end_tw = first_tw[shift]
        if arrival > end_tw:
            return False, -1, None, None
        start_srv = max(arrival, start_tw)
        wait = max(0, start_tw - arrival)
        total_wait += wait
        fix, var = first_params
        service = fix + (first_demand / var if var > 0 else 0)
        current_time = start_srv + service

        # between customers
        for i in range(len(route_list) - 1):
            a = route_list[i]
            b = route_list[i+1]
            a_idx, *_ = resolve(a)
            b_idx, b_demand, b_params, b_tw = resolve(b)
            travel = dist_matrix[a_idx, b_idx] / vel
            total_travel += travel
            arrival = current_time + travel
            start_tw, end_tw = b_tw[shift]
            if arrival > end_tw:
                return False, -1, None, None
            start_srv = max(arrival, start_tw)
            wait = max(0, start_tw - arrival)
            total_wait += wait
            fix, var = b_params
            service = fix + (b_demand / var if var > 0 else 0)
            current_time = start_srv + service

        # back to depot
        last_idx, *_ = resolve(route_list[-1])
        travel_back = depot_farm[depot_idx, last_idx] / vel
        total_travel += travel_back
        finish_time = current_time + travel_back
        if finish_time > 1440:
            return False, -1, None, None

        return True, finish_time, total_travel, total_wait

    # ============================================================
    best = None
    min_cost = float('inf')

    # ---- 1) Thử chèn vào các route hiện có ----
    # Khi chèn vào route hiện có, ta biết truck_id của route nên dùng truck_info = lookup
    trucks_by_id = {t['id']: t for t in problem_instance['fleet']['available_trucks']}

    for route_idx, route_info in enumerate(schedule_on_day):
        depot_idx, truck_id, custs, shift = route_info
        if shift == 'INTER-FACTORY': 
            continue

        # try each insertion position
        for pos in range(len(custs) + 1):
            new_route = custs[:pos] + [visit_id] + custs[pos:]
            truck_info = trucks_by_id.get(truck_id)
            feasible, finish, tot_travel, tot_wait = compute_route_metrics(depot_idx, new_route, shift, 0, truck_info)
            if not feasible:
                continue

            # variable cost per "distance" using truck type+region (fallback 1.0)
            var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
                (truck_info['type'], truck_info['region']), 1.0
            ) if truck_info else 1.0

            cost = tot_travel * var_cost_per_km + tot_wait * WAIT_COST_PER_MIN

            if cost < min_cost:
                min_cost = cost
                best = {'cost': cost, 'route_idx': route_idx, 'pos': pos, 'shift': shift, 'new_route_details': None}

    # ---- 2) Thử tạo 1 tuyến mới quanh visit ----
    # tìm depot gần nhất (như bạn đã làm)
    farm_idx, farm_details, farm_demand = _get_farm_info(visit_id, problem_instance)
    closest_depot_idx = int(np.argmin(problem_instance['distance_depots_farms'][:, farm_idx]))
    depot_region = problem_instance['facilities'][closest_depot_idx].get('region', None)

    # chọn trucks phù hợp region + accessibility + capacity
    suitable_trucks = []
    for t in problem_instance['fleet']['available_trucks']:
        if t.get('region') != depot_region: continue
        if t['capacity'] < farm_demand: continue
        t_idx = type_to_idx.get(t.get('type'))
        if t_idx is None: continue
        depot_acc = problem_instance['facilities'][closest_depot_idx].get('accessibility')
        farm_acc = farm_details.get('accessibility')
        depot_ok = (depot_acc is None or (len(depot_acc) > t_idx and depot_acc[t_idx] == 1))
        farm_ok = (farm_acc is None or (len(farm_acc) > t_idx and farm_acc[t_idx] == 1))
        if depot_ok and farm_ok:
            suitable_trucks.append(t)

    if suitable_trucks:
        # pick smallest-capacity truck that can serve (you used min capacity before)
        best_truck_for_new_route = min(suitable_trucks, key=lambda t: t['capacity'])
        for shift in ['AM', 'PM']:
            feasible, finish, tot_travel, tot_wait = compute_route_metrics(
                closest_depot_idx, [visit_id], shift, 0, best_truck_for_new_route
            )
            if not feasible:
                continue
            var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
                (best_truck_for_new_route['type'], best_truck_for_new_route['region']), 1.0
            )
            cost_new = tot_travel * var_cost_per_km + tot_wait * WAIT_COST_PER_MIN
            if cost_new < min_cost:
                min_cost = cost_new
                best = {
                    'cost': cost_new, 'route_idx': -1, 'pos': 0, 'shift': shift,
                    'new_route_details': (closest_depot_idx, best_truck_for_new_route['id'], [visit_id])
                }

    return best

# --- HÀM TẠO TUYẾN ĐƯỜNG MỚI (Không cần sửa, đã đúng) ---
def _create_new_route_for_farm(farm_id_to_insert, problem_instance, force_depot_idx=None):
    """Tạo một tuyến mới cho một farm."""
    facilities = problem_instance['facilities']
    dist_depot_data = problem_instance['distance_depots_farms']
    
    farm_idx, farm_details, farm_demand = _get_farm_info(farm_id_to_insert, problem_instance)
    
    depot_idx = force_depot_idx if force_depot_idx is not None else np.argmin(dist_depot_data[:, farm_idx])
    depot_region = facilities[depot_idx]['region']
    
    eligible_trucks = [
        t for t in problem_instance['fleet']['available_trucks'] 
        if t['region'] == depot_region and t['capacity'] >= farm_demand
    ]
    
    if not eligible_trucks: return None
        
    selected_truck = min(eligible_trucks, key=lambda t: t['capacity'])
    cost = dist_depot_data[depot_idx, farm_idx] * 2
    
    return cost, depot_idx, selected_truck['id'], [farm_id_to_insert]

# --- TOÁN TỬ REPAIR (Không cần sửa, logic đã đúng) ---
import copy

def best_insertion(current, random_state, **kwargs):
    """
    Sửa chữa theo "Đơn vị" khách hàng, tuân thủ checklist.
    """
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unserved_customers = list(kwargs['unvisited_customers'])
    
    failed_customers = []

    while unserved_customers:
        best_customer_to_insert = None
        best_package_of_insertions = None
        min_package_cost = float('inf')

        for customer_base_id in unserved_customers:
            _, farm_details, _ = _get_farm_info(customer_base_id, problem_instance)
            frequency = farm_details.get('frequency', 1)
            if frequency >= 1: visit_days = range(len(repaired.schedule))
            elif frequency == 0.5: visit_days = range(0, len(repaired.schedule), 2)
            else: visit_days = []
            
            # --- Logic xử lý Split Demand ---
            visits_per_day = defaultdict(list)
            virtual_map = problem_instance.get('virtual_split_farms', {})
            customer_has_split = False
            for key in virtual_map:
                if str(_clean_base_id(key)) == str(customer_base_id):
                    customer_has_split = True
                    day_match = re.search(r'_d(\d+)', key)
                    if day_match:
                        day_of_visit = int(day_match.group(1))
                        if day_of_visit in visit_days:
                            visits_per_day[day_of_visit].append(key)
            
            if not customer_has_split:
                for day in visit_days:
                    visits_per_day[day].append(customer_base_id)
            # --- Kết thúc logic ---

            current_package_insertions = {}
            is_package_feasible = True
            
            # Lặp qua các ngày cần phục vụ của khách hàng
            for day in visit_days:
                visits_to_insert_on_day = visits_per_day.get(day, [])
                if not visits_to_insert_on_day: is_package_feasible = False; break
                
                insertions_for_day = []
                temp_schedule_on_day = copy.deepcopy(repaired.schedule[day])
                
                for visit_id in visits_to_insert_on_day:
                    best_insertion = _find_best_insert_for_visit(temp_schedule_on_day, visit_id, problem_instance)
                    if best_insertion is None: is_package_feasible = False; break
                    
                    insertions_for_day.append({'visit_id': visit_id, **best_insertion})
                    
                    if best_insertion['route_idx'] == -1:
                        depot, truck, custs = best_insertion['new_route_details']
                        temp_schedule_on_day.append((depot, truck, custs, best_insertion['shift']))
                    else:
                        route_as_list = list(temp_schedule_on_day[best_insertion['route_idx']])
                        route_as_list[2].insert(best_insertion['pos'], visit_id)
                        temp_schedule_on_day[best_insertion['route_idx']] = tuple(route_as_list)

                if not is_package_feasible: break
                current_package_insertions[day] = insertions_for_day

            if is_package_feasible:
                package_cost = sum(ins['cost'] for day_ins in current_package_insertions.values() for ins in day_ins)
                if package_cost < min_package_cost:
                    min_package_cost = package_cost
                    best_customer_to_insert = customer_base_id
                    best_package_of_insertions = current_package_insertions

        if best_customer_to_insert:
            for day, insertions_details_list in best_package_of_insertions.items():
                for insertion_details in insertions_details_list:
                    visit_id = insertion_details['visit_id']
                    if insertion_details['route_idx'] == -1:
                        depot, truck, _ = insertion_details['new_route_details']
                        repaired.schedule[day].append((depot, truck, [visit_id], insertion_details['shift']))
                    else:
                        route_as_list = list(repaired.schedule[day][insertion_details['route_idx']])
                        route_as_list[2].insert(insertion_details['pos'], visit_id)
                        repaired.schedule[day][insertion_details['route_idx']] = tuple(route_as_list)
            
            unserved_customers.remove(best_customer_to_insert)
        else:
            failed_customers = unserved_customers
            print(f"!!! REPAIR FAILED: Không thể chèn các khách hàng: {failed_customers}")
            break

    return repaired, failed_customers

def _find_all_inserts_for_visit(schedule_on_day, visit_id, problem_instance):
    """
    Tìm TẤT CẢ các vị trí chèn khả thi cho một visit trong một ngày cụ thể
    và trả về một danh sách các phương án đã được sắp xếp theo chi phí.
    """
    all_insertions = []

    # --- PHẦN 1: THỬ CHÈN VÀO CÁC TUYẾN ĐƯỜNG HIỆN CÓ ---
    for route_idx, route_info in enumerate(schedule_on_day):
        # Không thể chèn vào tuyến vận chuyển liên kho
        if route_info[3] == 'INTER-FACTORY':
            continue
        
        # Thử chèn vào mọi vị trí trên tuyến
        for insert_pos in range(len(route_info[2]) + 1):
            is_feasible, cost_increase, _ = _check_insertion_feasibility(
                problem_instance,
                route_info,
                insert_pos,
                visit_id,
                route_info[3], # shift của tuyến hiện tại
                start_time=0
            )
            
            # Nếu vị trí chèn này là khả thi, thêm nó vào danh sách
            if is_feasible:
                all_insertions.append({
                    'cost': cost_increase,
                    'route_idx': route_idx,
                    'pos': insert_pos,
                    'shift': route_info[3],
                    'new_route_details': None
                })

    # --- PHẦN 2: THỬ TẠO MỘT TUYẾN ĐƯỜNG MỚI CHỈ CHỨA VISIT NÀY ---
    
    # Lấy thông tin cần thiết của visit để tìm xe và kho
    farm_idx, farm_details, farm_demand = _get_farm_info(visit_id, problem_instance)
    
    # Tìm depot gần nhất và các xe phù hợp
    facilities = problem_instance['facilities']
    closest_depot_idx = int(np.argmin(problem_instance['distance_depots_farms'][:, farm_idx]))
    depot_region = facilities[closest_depot_idx].get('region', None)

    # Lọc xe phù hợp
    type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
    suitable_trucks = []
    available_trucks = problem_instance['fleet']['available_trucks']

    for truck in available_trucks:
        if truck.get('region') != depot_region or truck['capacity'] < farm_demand:
            continue
        
        truck_type_idx = type_to_idx.get(truck['type'])
        if truck_type_idx is None:
            continue

        depot_access = facilities[closest_depot_idx].get('accessibility')
        farm_access = farm_details.get('accessibility')

        depot_ok = (depot_access is None or 
                    (len(depot_access) > truck_type_idx and depot_access[truck_type_idx] == 1))
        
        farm_ok = (farm_access is None or 
                   (len(farm_access) > truck_type_idx and farm_access[truck_type_idx] == 1))

        if depot_ok and farm_ok:
            suitable_trucks.append(truck)
    
    # Nếu có xe phù hợp để tạo tuyến mới
    if suitable_trucks:
        # Chọn xe có chi phí thấp nhất (ví dụ: xe có capacity nhỏ nhất mà vẫn đủ)
        best_truck_for_new_route = min(suitable_trucks, key=lambda t: t['capacity'])
        
        # Lấy các hằng số chi phí để tính chi phí của tuyến mới
        WAIT_COST_PER_MIN = 0.2
        var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
            (best_truck_for_new_route['type'], best_truck_for_new_route['region']), 1.0
        )
        
        # Thử tạo tuyến mới cho cả 2 ca (AM/PM)
        for shift in ['AM', 'PM']:
            # Gọi hàm tính toán để kiểm tra feasibility và lấy các thông số
            _, is_feasible, new_dist, new_wait, _ = _calculate_route_schedule_and_feasibility(
                closest_depot_idx,
                [visit_id],
                shift,
                0,
                problem_instance,
                best_truck_for_new_route
            )
            
            if is_feasible:
                # Chi phí của việc tạo tuyến mới = chi phí di chuyển + chi phí chờ
                cost_of_new_route = (new_dist * var_cost_per_km) + (new_wait * WAIT_COST_PER_MIN)
                
                # Thêm phương án "tạo tuyến mới" này vào danh sách
                all_insertions.append({
                    'cost': cost_of_new_route,
                    'route_idx': -1, # Mã hiệu cho việc tạo tuyến mới
                    'pos': 0,
                    'shift': shift,
                    'new_route_details': (closest_depot_idx, best_truck_for_new_route['id'], [visit_id])
                })
            
    # --- PHẦN 3: SẮP XẾP VÀ TRẢ VỀ KẾT QUẢ ---
    
    # Sắp xếp tất cả các phương án (cả chèn và tạo mới) từ tốt nhất đến tệ nhất
    all_insertions.sort(key=lambda x: x['cost'])
    
    return all_insertions

def _get_customer_schedule_pattern(customer_base_id, problem_instance, num_days):
    """Lấy "gói" lịch trình (các visit_id theo ngày) của một khách hàng."""
    _, farm_details, _ = _get_farm_info(customer_base_id, problem_instance)
    frequency = farm_details.get('frequency', 1)
    if frequency >= 1: visit_days = range(num_days)
    elif frequency == 0.5: visit_days = range(0, num_days, 2)
    else: visit_days = []
    
    visits_per_day = defaultdict(list)
    virtual_map = problem_instance.get('virtual_split_farms', {})
    customer_has_split = False
    for key in virtual_map:
        if str(_clean_base_id(key)) == str(customer_base_id):
            customer_has_split = True
            day_match = re.search(r'_d(\d+)', key)
            if day_match:
                day_of_visit = int(day_match.group(1))
                if day_of_visit in visit_days:
                    visits_per_day[day_of_visit].append(key)
    
    if not customer_has_split:
        for day in visit_days:
            visits_per_day[day].append(customer_base_id)
            
    return visits_per_day

def regret_insertion(current, random_state, **kwargs):
    """
    BẮT BUỘC: Sửa chữa bằng cách ưu tiên khách hàng có "sự hối tiếc" cao nhất.
    """
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unserved_customers = list(kwargs['unvisited_customers'])
    
    failed_customers = []
    
    # K-value cho Regret-K
    K = kwargs.get('k_regret', 3) 

    while unserved_customers:
        customer_regret_options = []

        # 1. Với mỗi khách hàng, tính toán "regret"
        for customer_base_id in unserved_customers:
            visits_per_day = _get_customer_schedule_pattern(customer_base_id, problem_instance, len(repaired.schedule))
            
            all_package_options = []
            
            # Để tính regret cho cả gói, ta cần một cách tiếp cận đơn giản hơn:
            # Tính tổng chi phí của các lựa chọn tốt nhất cho mỗi ngày
            # và tổng chi phí của các lựa chọn tốt thứ hai cho mỗi ngày, v.v.
            
            package_costs = []
            is_package_possible = True
            
            # Tìm tất cả các phương án chèn cho từng visit trong gói
            all_visit_insertions = {}
            for day, visits in visits_per_day.items():
                for visit_id in visits:
                    inserts = _find_all_inserts_for_visit(repaired.schedule[day], visit_id, problem_instance)
                    if not inserts:
                        is_package_possible = False
                        break
                    all_visit_insertions[(day, visit_id)] = inserts
                if not is_package_possible: break
            
            if not is_package_possible:
                continue # Bỏ qua khách hàng này nếu một visit không thể chèn được

            # Đây là một heuristic đơn giản để tính regret cho cả gói:
            # Tính regret cho từng visit và cộng dồn lại
            total_regret = 0
            best_package_cost = 0
            best_package_details = {}

            for (day, visit_id), inserts in all_visit_insertions.items():
                best_insert = inserts[0]
                best_package_cost += best_insert['cost']
                best_package_details[(day, visit_id)] = best_insert
                
                regret_for_visit = 0
                if len(inserts) >= K:
                    for i in range(1, K):
                        regret_for_visit += (inserts[i]['cost'] - best_insert['cost'])
                elif len(inserts) > 1:
                    regret_for_visit += (inserts[1]['cost'] - best_insert['cost'])
                
                total_regret += regret_for_visit

            customer_regret_options.append({
                'regret': total_regret,
                'customer': customer_base_id,
                'cost': best_package_cost,
                'package': best_package_details
            })

        if not customer_regret_options:
            failed_customers = unserved_customers
            break

        # 2. Chọn khách hàng có regret cao nhất
        best_regret_option = max(customer_regret_options, key=lambda x: x['regret'])
        
        # 3. Thực hiện chèn khách hàng đó vào vị trí tốt nhất
        customer_to_insert = best_regret_option['customer']
        package_to_insert = best_regret_option['package']
        
        for (day, visit_id), insertion_details in package_to_insert.items():
            if insertion_details['route_idx'] == -1:
                depot, truck, _ = insertion_details['new_route_details']
                repaired.schedule[day].append((depot, truck, [visit_id], insertion_details['shift']))
            else:
                route_as_list = list(repaired.schedule[day][insertion_details['route_idx']])
                route_as_list[2].insert(insertion_details['pos'], visit_id)
                repaired.schedule[day][insertion_details['route_idx']] = tuple(route_as_list)
        
        unserved_customers.remove(customer_to_insert)

    return repaired, failed_customers

#! Mấy repairs dưới chưa đổi theo yếu tố multi-trip, cần sửa lại sau



def _find_k_best_package_insertions(k, customer_base_id, schedule, problem_instance):
    """
    Hàm trợ giúp cực kỳ phức tạp: Tìm K phương án chèn "gói" tốt nhất cho một khách hàng.
    Trả về một danh sách các "gói", mỗi gói là một dict và có tổng chi phí.
    """
    num_days = len(schedule)
    visits_per_day = _get_customer_schedule_pattern(customer_base_id, problem_instance, num_days)
    
    # 1. Tìm tất cả các lựa chọn chèn cho mỗi visit riêng lẻ
    options_per_visit = {}
    for day, visits in visits_per_day.items():
        for visit_id in visits:
            inserts = _find_all_inserts_for_visit(schedule[day], visit_id, problem_instance)
            if not inserts:
                return [] # Nếu một visit không thể chèn, cả gói thất bại
            options_per_visit[(day, visit_id)] = inserts

    # 2. Xây dựng các "gói" hoàn chỉnh bằng cách kết hợp các lựa chọn
    # Đây là một bài toán tổ hợp. Để đơn giản, ta chỉ xét một vài kết hợp đầu tiên.
    
    # Lấy danh sách các lựa chọn cho từng visit
    list_of_options = list(options_per_visit.values())
    visit_ids = list(options_per_visit.keys())
    
    package_options = []
    
    # Dùng itertools.product để tạo ra các tổ hợp
    # Cảnh báo: có thể rất chậm nếu K lớn hoặc có nhiều visit
    # Giới hạn số lựa chọn cho mỗi visit để giảm độ phức tạp
    limited_list_of_options = [opts[:k] for opts in list_of_options]
    
    for combo in itertools.product(*limited_list_of_options):
        package_cost = sum(insert['cost'] for insert in combo)
        package_details = {visit_ids[i]: combo[i] for i in range(len(visit_ids))}
        package_options.append({'cost': package_cost, 'details': package_details})

    # Sắp xếp các gói theo chi phí và trả về K gói tốt nhất
    package_options.sort(key=lambda x: x['cost'])
    return package_options[:k]


def _regret_k_insertion(current, random_state, k_regret, **kwargs):
    """
    BẮT BUỘC: Sửa chữa bằng cách ưu tiên khách hàng có "sự hối tiếc" (regret) cao nhất.
    """
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unserved_customers = list(kwargs['unvisited_customers'])
    
    failed_customers = []

    while unserved_customers:
        customer_regret_options = []

        # 1. Với mỗi khách hàng, tính toán "regret"
        for customer_base_id in unserved_customers:
            # Tìm K phương án chèn "gói" tốt nhất cho khách hàng này
            package_options = _find_k_best_package_insertions(k_regret, customer_base_id, repaired.schedule, problem_instance)

            if not package_options:
                continue # Bỏ qua nếu không có phương án chèn nào khả thi

            best_package = package_options[0]
            regret_value = 0
            
            # Tính regret bằng tổng chênh lệch chi phí so với phương án tốt nhất
            for i in range(1, len(package_options)):
                regret_value += (package_options[i]['cost'] - best_package['cost'])
            
            customer_regret_options.append({
                'regret': regret_value,
                'customer': customer_base_id,
                'best_package': best_package
            })

        if not customer_regret_options:
            failed_customers = unserved_customers
            break

        # 2. Chọn khách hàng có regret cao nhất
        best_regret_option = max(customer_regret_options, key=lambda x: x['regret'])
        
        # 3. Thực hiện chèn khách hàng đó vào vị trí tốt nhất của nó
        customer_to_insert = best_regret_option['customer']
        package_to_insert = best_regret_option['best_package']['details']
        
        # Tạo một schedule tạm thời để chèn, tránh xung đột
        temp_schedule = copy.deepcopy(repaired.schedule)
        
        for (day, visit_id), insertion_details in package_to_insert.items():
            if insertion_details['route_idx'] == -1:
                depot, truck, _ = insertion_details['new_route_details']
                temp_schedule[day].append((depot, truck, [visit_id], insertion_details['shift']))
            else:
                route_as_list = list(temp_schedule[day][insertion_details['route_idx']])
                route_as_list[2].insert(insertion_details['pos'], visit_id)
                temp_schedule[day][insertion_details['route_idx']] = tuple(route_as_list)
        
        repaired.schedule = temp_schedule
        unserved_customers.remove(customer_to_insert)

    return repaired, failed_customers


def regret_insertion(current, random_state, **kwargs):
    """Toán tử Regret-K với K=3 (một lựa chọn phổ biến)."""
    return _regret_k_insertion(current, random_state, k_regret=3, **kwargs)

def regret2_insertion(current, random_state, **kwargs):
    return _regret_k_insertion(current, random_state, k_regret=2, **kwargs)

def regret3_insertion(current, random_state, **kwargs):
    return _regret_k_insertion(current, random_state, k_regret=3, **kwargs)

def cheapest_feasible_insertion(current, random_state, **kwargs):
    # Logic của cheapest_feasible rất giống best_insertion, chỉ khác ở cách lặp
    # Thay vì tìm vị trí tốt nhất cho tất cả rồi chọn 1, nó tìm và chèn ngay lập tức
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unvisited_customers = list(kwargs['unvisited_customers'])

    # Lặp lại cho đến khi không còn khách hàng nào để chèn
    inserted_in_this_pass = True
    while inserted_in_this_pass:
        inserted_in_this_pass = False
        best_cost_this_pass = float('inf')
        best_details_this_pass = None
        farm_to_insert_this_pass = None
        
        if not unvisited_customers: break

        for farm_id in unvisited_customers:
            insertions = _get_all_insertions_for_farm(repaired.schedule, farm_id, problem_instance, random_state)
            if insertions:
                best_for_farm = min(insertions, key=lambda x: x[0])
                if best_for_farm[0] < best_cost_this_pass:
                    best_cost_this_pass = best_for_farm[0]
                    best_details_this_pass = best_for_farm
                    farm_to_insert_this_pass = farm_id
        
        if farm_to_insert_this_pass:
            cost, day_idx, route_idx, pos, shift, truck_id = best_details_this_pass
            repaired.schedule[day_idx][route_idx][2].insert(pos, farm_to_insert_this_pass)
            unvisited_customers.remove(farm_to_insert_this_pass)
            inserted_in_this_pass = True
            
    # Xử lý các khách hàng còn lại không thể chèn vào tuyến có sẵn
    for farm_id in unvisited_customers:
        new_route_info = _create_new_route_for_farm(farm_id, problem_instance)
        if new_route_info:
            cost, depot_idx, truck_id, cust_list = new_route_info
            random_day = random_state.choice(list(repaired.schedule.keys()))
            repaired.schedule[random_day].append([depot_idx, truck_id, cust_list])

    return repaired


def random_feasible_insertion(current, random_state, **kwargs):
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unvisited_customers = list(kwargs['unvisited_customers'])
    random_state.shuffle(unvisited_customers)

    for farm_id in unvisited_customers:
        insertions = _get_all_insertions_for_farm(repaired.schedule, farm_id, problem_instance, random_state)
        
        if insertions:
            # Chọn một vị trí chèn ngẫu nhiên từ các vị trí khả thi
            chosen_insertion = random_state.choice(insertions)
            cost, day_idx, route_idx, pos, shift, truck_id = chosen_insertion
            repaired.schedule[day_idx][route_idx][2].insert(pos, farm_id)
        else:
            # Nếu không chèn được, tạo tuyến mới
            new_route_info = _create_new_route_for_farm(farm_id, problem_instance)
            if new_route_info:
                cost, depot_idx, truck_id, cust_list = new_route_info
                random_day = random_state.choice(list(repaired.schedule.keys()))
                repaired.schedule[random_day].append([depot_idx, truck_id, cust_list])
                
    return repaired

def regret_insertion(current, random_state, **kwargs):
    """
    Toán tử sửa chữa Regret Insertion. 
    Đây là tên gọi phổ biến cho Regret-2, so sánh giữa lựa chọn tốt nhất và tốt thứ hai.
    """
    return _regret_k_insertion(current, random_state, k_regret=2, **kwargs)