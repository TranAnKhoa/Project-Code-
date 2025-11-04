import copy
import random
import numpy as np
import re
from collections import defaultdict

# ==============================================================================
# HÀM TIỆN ÍCH CHUNG
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
            raise KeyError(f"DestroyOp: Không thể tìm thấy Farm ID '{base_id}' (từ '{farm_id}') trong map.")
            
    farm_details = farms[farm_idx]
    
    if farm_id in virtual_map:
        demand = virtual_map[farm_id]['portion']
    else:
        demand = farm_details['demand']
        
    return farm_idx, farm_details, demand

# --- HÀM TRỢ GIÚP CHO VIỆC XÓA THEO PATTERN ---
def _remove_customer_pattern_from_schedule(schedule, customer_base_id):
    """
    Xóa TOÀN BỘ các lần ghé thăm của một khách hàng gốc khỏi lịch trình.
    """
    new_schedule = {day: [] for day in schedule}
    
    visits_to_remove = set()
    for day_routes in schedule.values():
        for _, _, customer_list, _ in day_routes:
            for fid in customer_list:
                if str(_clean_base_id(fid)) == str(customer_base_id):
                    visits_to_remove.add(fid)

    for day, routes in schedule.items():
        for route_info in routes:
            depot_idx, truck_id, customer_list, shift = route_info
            
            new_customer_list = [cid for cid in customer_list if cid not in visits_to_remove]
            
            if new_customer_list or shift == 'INTER-FACTORY':
                new_schedule[day].append((depot_idx, truck_id, new_customer_list, shift))
    
    return new_schedule

# ==============================================================================
# CÁC TOÁN TỬ PHÁ HỦY (VIẾT LẠI THEO YÊU CẦU BẮT BUỘC)
# ==============================================================================

def random_removal(current, random_state, **kwargs):
    """
    Phá hủy theo "Đơn vị" là toàn bộ lịch trình của một khách hàng.
    """
    destroyed = copy.deepcopy(current)
    
    all_base_customers = set()
    for day_routes in destroyed.schedule.values():
        for _, _, customer_list, shift in day_routes:
            if shift != 'INTER-FACTORY':
                for fid in customer_list:
                    all_base_customers.add(_clean_base_id(fid))
    
    if not all_base_customers:
        return destroyed, []
    
    all_base_customers = list(all_base_customers)
    
    num_to_remove = kwargs.get('num_to_remove', max(1, int(len(all_base_customers) * 0.1)))
    num_to_remove = min(num_to_remove, len(all_base_customers))

    customers_to_remove = random.sample(all_base_customers, num_to_remove)
    
    for base_id in customers_to_remove:
        destroyed.schedule = _remove_customer_pattern_from_schedule(destroyed.schedule, base_id)
    
    return destroyed, customers_to_remove


def worst_removal(current, random_state, **kwargs):
    """
    Phá hủy lịch trình của các khách hàng có chi phí "đắt" nhất.
    """
    destroyed = copy.deepcopy(current)
    problem_instance = destroyed.problem_instance
    dist_depot_farm = problem_instance['distance_depots_farms']
    dist_farm_farm = problem_instance['distance_matrix_farms']

    customer_costs = defaultdict(float)
    for day_idx, routes in destroyed.schedule.items():
        for route_info in routes:
            depot_idx, _, customer_list, shift = route_info
            if not customer_list or shift == 'INTER-FACTORY': continue

            for i, farm_id in enumerate(customer_list):
                farm_idx, _, _ = _get_farm_info(farm_id, problem_instance)
                
                cost_saving = 0
                if len(customer_list) == 1:
                    cost_saving = dist_depot_farm[depot_idx, farm_idx] * 2
                elif i == 0:
                    next_farm_idx, _, _ = _get_farm_info(customer_list[i+1], problem_instance)
                    cost_saving = (dist_depot_farm[depot_idx, farm_idx] + dist_farm_farm[farm_idx, next_farm_idx] - dist_depot_farm[depot_idx, next_farm_idx])
                elif i == len(customer_list) - 1:
                    prev_farm_idx, _, _ = _get_farm_info(customer_list[i-1], problem_instance)
                    cost_saving = (dist_farm_farm[prev_farm_idx, farm_idx] + dist_depot_farm[depot_idx, farm_idx] - dist_depot_farm[depot_idx, prev_farm_idx])
                else:
                    prev_farm_idx, _, _ = _get_farm_info(customer_list[i-1], problem_instance)
                    next_farm_idx, _, _ = _get_farm_info(customer_list[i+1], problem_instance)
                    cost_saving = (dist_farm_farm[prev_farm_idx, farm_idx] + dist_farm_farm[farm_idx, next_farm_idx] - dist_farm_farm[prev_farm_idx, next_farm_idx])
                
                customer_costs[_clean_base_id(farm_id)] += cost_saving
    
    if not customer_costs:
        return destroyed, []
        
    num_to_remove = kwargs.get('num_to_remove', max(1, int(len(customer_costs) * 0.2)))
    num_to_remove = min(num_to_remove, len(customer_costs))

    sorted_customers = sorted(customer_costs.items(), key=lambda item: item[1], reverse=True)
    
    customers_to_remove = []
    for _ in range(num_to_remove):
        if not sorted_customers: break
        rand_val = random_state.random()
        idx_to_remove = int(len(sorted_customers) * (rand_val ** 4))
        
        customer_id, cost = sorted_customers.pop(idx_to_remove)
        customers_to_remove.append(customer_id)
    
    for base_id in customers_to_remove:
        destroyed.schedule = _remove_customer_pattern_from_schedule(destroyed.schedule, base_id)

    return destroyed, customers_to_remove
# --- CÁC TOÁN TỬ KHÁC ---
# (Các toán tử này bây giờ cũng cần được viết lại theo cùng một logic)
def shaw_removal(current, random_state, **kwargs):
    """
    Xóa một nhóm các khách hàng "liên quan" đến nhau.
    """
    destroyed = copy.deepcopy(current)
    problem = destroyed.problem_instance
    
    # Lấy danh sách khách hàng gốc và các visit
    all_base_customers = set()
    all_visits = []
    for day, routes in destroyed.schedule.items():
        for _, _, cust_list, shift in routes:
            if shift != 'INTER-FACTORY':
                for fid in cust_list:
                    all_base_customers.add(_clean_base_id(fid))
                    all_visits.append({'id': fid, 'day': day})

    if not all_base_customers:
        return destroyed, []
        
    num_to_remove = kwargs.get('num_to_remove', max(1, int(len(all_base_customers) * 0.15)))

    # Chọn ngẫu nhiên một visit làm "hạt giống"
    seed_visit = random.choice(all_visits)
    removed_customers_base = {_clean_base_id(seed_visit['id'])}
    
    # Vòng lặp để tìm các visit liên quan
    while len(removed_customers_base) < num_to_remove:
        # Lấy một visit ngẫu nhiên từ những visit đã bị xóa để so sánh
        last_removed_id = random.choice(list(removed_customers_base))
        
        relatedness = []
        for other_cust_base_id in all_base_customers:
            if other_cust_base_id in removed_customers_base:
                continue
            
            # Tính độ liên quan (đơn giản hóa: chỉ dựa trên khoảng cách)
            idx1, _, _ = _get_farm_info(last_removed_id, problem)
            idx2, _, _ = _get_farm_info(other_cust_base_id, problem)
            distance = problem['distance_matrix_farms'][idx1, idx2]
            
            # (Có thể thêm các yếu tố khác: chênh lệch time window, chênh lệch demand...)
            relatedness_score = 1 / (distance + 1e-5) # Càng gần, điểm càng cao
            relatedness.append((relatedness_score, other_cust_base_id))

        if not relatedness: break
            
        relatedness.sort(key=lambda x: x[0], reverse=True)
        
        # Chọn khách hàng liên quan nhất (có yếu tố ngẫu nhiên)
        rand_val = random_state.random()
        idx_to_add = int(len(relatedness) * (rand_val ** 2))
        
        customer_to_add = relatedness[idx_to_add][1]
        removed_customers_base.add(customer_to_add)

    # Xóa toàn bộ pattern của các khách hàng đã chọn
    for base_id in removed_customers_base:
        destroyed.schedule = _remove_customer_pattern_from_schedule(destroyed.schedule, base_id)
        
    return destroyed, list(removed_customers_base)

def truck_day_removal(current, random_state, **kwargs):
    # Logic này phức tạp hơn vì nó xóa các xe, không phải khách hàng.
    # Tạm thời, để đảm bảo tính nhất quán, hãy để nó gọi random_removal.
    # Khi bạn sẵn sàng, bạn có thể viết lại nó để trả về danh sách khách hàng gốc.
    return random_removal(current, random_state, **kwargs)

def shaw_distance_removal(current, random_state, **kwargs):
    # Cần viết lại hoàn toàn theo logic "đơn vị khách hàng"
    return random_removal(current, random_state, **kwargs)

# Các hàm còn lại trỏ đến các phiên bản đã hoạt động
relatedness_removal = shaw_distance_removal
cost_outlier_removal = worst_removal
path_removal = random_removal
sequence_removal = random_removal
history_biased_removal = random_removal
two_route_exchange = random_removal
depot_shift_removal = random_removal
load_balancing_removal = random_removal