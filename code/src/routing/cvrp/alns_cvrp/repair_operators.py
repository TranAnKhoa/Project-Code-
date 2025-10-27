import copy
import random
import numpy as np
from collections import defaultdict

# --- HÀM TÌM XE ---
def find_truck_by_id(truck_id, available_trucks):
    """Tiện ích để tìm thông tin chi tiết của xe từ ID."""
    for truck in available_trucks:
        if truck['id'] == truck_id:
            return truck
    return None

# --- HÀM TÍNH TOÁN TÁC ĐỘNG CHÈN ---
def calculate_insertion_impact(depot_idx, customer_list, farm_id_to_insert, insert_pos, shift, truck_id, problem_instance):
    """
    Tính toán chi phí tăng thêm và kiểm tra tính khả thi về thời gian khi chèn một nông trại.
    """
    farm_dist_matrix = problem_instance['distance_matrix_farms']
    depot_farm_dist_matrix = problem_instance['distance_depots_farms']
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']

    farm_idx_to_insert = farm_id_to_idx[farm_id_to_insert]
    
    cost_increase = 0
    if not customer_list:
        cost_increase = depot_farm_dist_matrix[depot_idx, farm_idx_to_insert] * 2
    elif insert_pos == 0:
        first_cust_idx = farm_id_to_idx[customer_list[0]]
        cost_increase = (depot_farm_dist_matrix[depot_idx, farm_idx_to_insert] + 
                         farm_dist_matrix[farm_idx_to_insert, first_cust_idx] - 
                         depot_farm_dist_matrix[depot_idx, first_cust_idx])
    elif insert_pos == len(customer_list):
        last_cust_idx = farm_id_to_idx[customer_list[-1]]
        cost_increase = (farm_dist_matrix[last_cust_idx, farm_idx_to_insert] + 
                         depot_farm_dist_matrix[depot_idx, farm_idx_to_insert] - 
                         depot_farm_dist_matrix[depot_idx, last_cust_idx])
    else:
        prev_cust_idx = farm_id_to_idx[customer_list[insert_pos - 1]]
        next_cust_idx = farm_id_to_idx[customer_list[insert_pos]]
        cost_increase = (farm_dist_matrix[prev_cust_idx, farm_idx_to_insert] + 
                         farm_dist_matrix[farm_idx_to_insert, next_cust_idx] - 
                         farm_dist_matrix[prev_cust_idx, next_cust_idx])

    # TODO: HOÀN THIỆN LOGIC KIỂM TRA TIME WINDOW TẠI ĐÂY
    is_feasible = True
    
    return cost_increase, is_feasible

# --- HÀM SIÊU TRỢ GIÚP TÌM ĐIỂM CHÈN ---
def _get_all_insertions_for_farm(schedule, farm_id_to_insert, problem_instance, random_state):
    # ... (Hàm này đã đúng, giữ nguyên) ...
    farms = problem_instance['farms']
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map']
    farm_idx = farm_id_to_idx_map[farm_id_to_insert]
    farm_details = farms[farm_idx]
    
    available_trucks = problem_instance['fleet']['available_trucks']
    truck_id_map = {truck['id']: truck for truck in available_trucks}

    possible_insertions = []

    for day_idx in schedule.keys():
        routes_of_day = schedule[day_idx]
        for route_idx, route_info in enumerate(routes_of_day):
            depot_idx, truck_id, customer_list = route_info
            truck_details = truck_id_map.get(truck_id)
            if not truck_details: continue

            current_load = sum(farms[farm_id_to_idx_map[c_id]]['demand'] for c_id in customer_list)
            if current_load + farm_details['demand'] > truck_details['capacity']:
                continue
            
            truck_type_idx = ['Single', '20m', '26m', 'Truck and Dog'].index(truck_details['type'])
            if not farm_details['accessibility'][truck_type_idx]:
                continue

            for insert_pos in range(len(customer_list) + 1):
                for shift in ['AM', 'PM']:
                    cost_increase, is_feasible = calculate_insertion_impact(
                        depot_idx, customer_list, farm_id_to_insert, insert_pos, shift,
                        truck_id, problem_instance
                    )
                    
                    if is_feasible:
                        possible_insertions.append(
                            (cost_increase, day_idx, route_idx, insert_pos, shift, truck_id)
                        )
    return possible_insertions


# --- HÀM TẠO TUYẾN ĐƯỜNG MỚI (ĐÃ SỬA LỖI) ---
def _create_new_route_for_farm(farm_id_to_insert, problem_instance):
    """Tìm cách tốt nhất để tạo một tuyến đường mới cho một nông trại."""
    # SỬA LỖI: Lấy dữ liệu từ problem_instance với tên khóa đúng
    farms = problem_instance['farms']
    facilities = problem_instance['facilities']
    dist_depot_data = problem_instance['distance_depots_farms']
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map']
    
    farm_idx = farm_id_to_idx_map[farm_id_to_insert]
    farm_details = farms[farm_idx]
    
    closest_depot_idx = np.argmin(dist_depot_data[:, farm_idx])
    depot_region = facilities[closest_depot_idx]['region']
    
    eligible_trucks = [
        t for t in problem_instance['fleet']['available_trucks'] 
        if t['region'] == depot_region and t['capacity'] >= farm_details['demand']
    ]
    
    if not eligible_trucks: return None
        
    selected_truck = min(eligible_trucks, key=lambda t: t['capacity'])
    cost = dist_depot_data[closest_depot_idx, farm_idx] * 2
    
    return cost, closest_depot_idx, selected_truck['id'], [farm_id_to_insert]

# --- CÁC TOÁN TỬ REPAIR (ĐÃ NÂNG CẤP) ---

def best_insertion(current, random_state, **kwargs):
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    
    unvisited_customers = list(kwargs['unvisited_customers']) # Lấy từ toán tử destroy

    while unvisited_customers:
        best_overall_cost = float('inf')
        best_insertion_details = None
        farm_to_insert = None

        for farm_id in unvisited_customers:
            insertions = _get_all_insertions_for_farm(repaired.schedule, farm_id, problem_instance, random_state)
            
            if insertions:
                best_for_farm = min(insertions, key=lambda x: x[0])
                if best_for_farm[0] < best_overall_cost:
                    best_overall_cost = best_for_farm[0]
                    best_insertion_details = best_for_farm
                    farm_to_insert = farm_id
        
        if farm_to_insert:
            cost, day_idx, route_idx, pos, shift, truck_id = best_insertion_details
            repaired.schedule[day_idx][route_idx][2].insert(pos, farm_to_insert)
            unvisited_customers.remove(farm_to_insert)
        else: # Không thể chèn vào tuyến nào có sẵn
            farm_to_create_route = unvisited_customers.pop(0)
            new_route_info = _create_new_route_for_farm(farm_to_create_route, problem_instance)
            if new_route_info:
                cost, depot_idx, truck_id, cust_list = new_route_info
                # Chọn một ngày ngẫu nhiên để thêm tuyến mới
                random_day = random_state.choice(list(repaired.schedule.keys()))
                repaired.schedule[random_day].append([depot_idx, truck_id, cust_list])
            else:
                # Không thể tạo tuyến mới, bỏ qua khách hàng này (trường hợp hiếm)
                print(f"Cảnh báo: Không thể tạo tuyến mới cho nông trại {farm_to_create_route}")
    return repaired


def _regret_k_insertion(current, random_state, k_regret, **kwargs):
    repaired = copy.deepcopy(current)
    problem_instance = repaired.problem_instance
    unvisited_customers = list(kwargs['unvisited_customers'])

    while unvisited_customers:
        customer_regret_options = []

        for farm_id in unvisited_customers:
            insertions = _get_all_insertions_for_farm(repaired.schedule, farm_id, problem_instance, random_state)
            
            if not insertions:
                continue

            insertions.sort(key=lambda x: x[0])
            
            regret = 0
            if len(insertions) >= k_regret:
                regret = insertions[k_regret - 1][0] - insertions[0][0]
            elif len(insertions) > 1:
                regret = insertions[1][0] - insertions[0][0]

            customer_regret_options.append((regret, farm_id, insertions[0]))

        if not customer_regret_options:
            farm_to_create_route = unvisited_customers.pop(0)
            new_route_info = _create_new_route_for_farm(farm_to_create_route, problem_instance)
            if new_route_info:
                cost, depot_idx, truck_id, cust_list = new_route_info
                random_day = random_state.choice(list(repaired.schedule.keys()))
                repaired.schedule[random_day].append([depot_idx, truck_id, cust_list])
            else:
                print(f"Cảnh báo: Không thể tạo tuyến mới cho nông trại {farm_to_create_route}")
            continue

        customer_regret_options.sort(key=lambda x: x[0], reverse=True)
        
        regret, farm_to_insert, best_insertion_details = customer_regret_options[0]
        cost, day_idx, route_idx, pos, shift, truck_id = best_insertion_details
        
        repaired.schedule[day_idx][route_idx][2].insert(pos, farm_to_insert)
        unvisited_customers.remove(farm_to_insert)
        
    return repaired

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