import copy
import random
import numpy as np
from routing.cvrp.alns_cvrp.cvrp_helper_functions import determine_nr_nodes_to_remove, NormalizeData

# --- HÀM TRỢ GIÚP CHUNG ---
def _flatten_schedule_to_customer_list(schedule):
    """Lấy danh sách tất cả các customer_id đang được phục vụ trong lịch trình."""
    return [cust_id for day in schedule for route_info in schedule[day] for cust_id in route_info[2]]

def _remove_customers_from_schedule(schedule, customers_to_remove):
    """Xóa một danh sách khách hàng khỏi lịch trình và trả về lịch trình mới."""
    customers_to_remove_set = set(customers_to_remove)
    new_schedule = {day: [] for day in schedule}

    for day, routes in schedule.items():
        for depot_idx, truck_id, customer_list in routes:
            # Tạo một customer_list mới không chứa các khách hàng cần xóa
            new_customer_list = [cid for cid in customer_list if cid not in customers_to_remove_set]
            
            # Chỉ giữ lại tuyến đường nếu nó không rỗng
            if new_customer_list:
                new_schedule[day].append((depot_idx, truck_id, new_customer_list))
    return new_schedule

# --- CÁC TOÁN TỬ DESTROY ĐÃ ĐƯỢC NÂNG CẤP ---

def random_removal(current, random_state, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    nr_nodes_to_remove = kwargs.get('nr_nodes_to_remove', determine_nr_nodes_to_remove(current.nb_customers))
    
    visited_customers = _flatten_schedule_to_customer_list(destroyed_solution.schedule)
    
    if not visited_customers:
        return destroyed_solution, []

    num_to_remove_actual = min(nr_nodes_to_remove, len(visited_customers))
    customers_to_remove = random_state.choice(visited_customers, num_to_remove_actual, replace=False).tolist()
    
    destroyed_solution.schedule = _remove_customers_from_schedule(destroyed_solution.schedule, customers_to_remove)
    
    # Trả về solution đã bị phá hủy và danh sách khách hàng đã bị xóa
    return destroyed_solution, customers_to_remove

def shaw_distance_removal(current, random_state, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    nr_nodes_to_remove = kwargs.get('nr_nodes_to_remove', determine_nr_nodes_to_remove(current.nb_customers))
    prob = kwargs.get('prob', 5) # Lấy prob từ kwargs nếu có

    visited_customers = _flatten_schedule_to_customer_list(destroyed_solution.schedule)
    if not visited_customers:
        return destroyed_solution, []

    # Chọn ngẫu nhiên một khách hàng làm mồi
    seed_customer = random_state.choice(visited_customers)
    customers_to_remove = [seed_customer]
    
    # Duyệt để tìm các khách hàng liên quan
    while len(customers_to_remove) < nr_nodes_to_remove:
        # Lấy khách hàng cuối cùng đã thêm vào làm mồi cho vòng lặp tiếp theo
        current_seed = customers_to_remove[-1]
        
        # Lấy danh sách các khách hàng còn lại
        remaining_customers = [c for c in visited_customers if c not in customers_to_remove]
        if not remaining_customers: break

        relatedness_scores = []
        for other_cust in remaining_customers:
            # Tính toán độ tương quan (ở đây là khoảng cách)
            dist = current.dist_matrix_data[current_seed - 1, other_cust - 1]
            relatedness_scores.append((dist, other_cust))
        
        relatedness_scores.sort(key=lambda x: x[0]) # Sắp xếp theo khoảng cách tăng dần
        
        # Áp dụng lựa chọn ngẫu nhiên (deterministic randomness)
        idx_to_pick = int(len(relatedness_scores) ** random_state.random())
        
        customer_to_add = relatedness_scores[idx_to_pick][1]
        customers_to_remove.append(customer_to_add)

    destroyed_solution.schedule = _remove_customers_from_schedule(destroyed_solution.schedule, customers_to_remove)
    return destroyed_solution, customers_to_remove


def worst_removal(current, random_state, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    nr_nodes_to_remove = kwargs.get('nr_nodes_to_remove', determine_nr_nodes_to_remove(current.nb_customers))

    # Tính toán chi phí "tiết kiệm được" nếu xóa mỗi khách hàng
    removal_candidates = [] # (cost_saving, customer_id)
    
    for day, routes in destroyed_solution.schedule.items():
        for depot_idx, truck_id, customer_list in routes:
            for i, cust_id in enumerate(customer_list):
                cust_idx = cust_id - 1
                
                # Tính chi phí hiện tại của việc ghé thăm khách hàng này
                prev_node_idx = customer_list[i-1] - 1 if i > 0 else depot_idx
                next_node_idx = customer_list[i+1] - 1 if i < len(customer_list)-1 else depot_idx
                
                cost_with_cust = (current.dist_matrix_data[prev_node_idx, cust_idx] + 
                                  current.dist_matrix_data[cust_idx, next_node_idx])
                
                # Tính chi phí nếu bỏ qua khách hàng này
                cost_without_cust = current.dist_matrix_data[prev_node_idx, next_node_idx]
                
                cost_saving = cost_with_cust - cost_without_cust
                removal_candidates.append((cost_saving, cust_id))
    
    if not removal_candidates:
        return destroyed_solution, []

    # Sắp xếp theo chi phí tiết kiệm được giảm dần (xóa cái nào lợi nhất)
    removal_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Áp dụng lựa chọn ngẫu nhiên
    num_to_remove_actual = min(nr_nodes_to_remove, len(removal_candidates))
    customers_to_remove = []
    
    while len(customers_to_remove) < num_to_remove_actual and removal_candidates:
        idx_to_pick = int(len(removal_candidates) ** random_state.random())
        cust_to_remove = removal_candidates.pop(idx_to_pick)[1]
        customers_to_remove.append(cust_to_remove)
        # Loại bỏ các bản sao của khách hàng này khỏi danh sách ứng viên
        removal_candidates = [cand for cand in removal_candidates if cand[1] != cust_to_remove]

    destroyed_solution.schedule = _remove_customers_from_schedule(destroyed_solution.schedule, customers_to_remove)
    return destroyed_solution, customers_to_remove

# Các hàm khác có thể được cập nhật tương tự
# Ví dụ: cluster_removal
def cluster_removal(current, random_state, **kwargs):
    # Logic tương tự shaw_removal, chỉ khác ở chỗ nó xóa một cụm gần nhau
    return shaw_distance_removal(current, random_state, **kwargs)


# Các toán tử khác phức tạp hơn (liên quan đến graph, history...) sẽ cần
# điều chỉnh cách chúng tính toán "chi phí" hoặc "điểm số". 
# Dưới đây là các phiên bản đơn giản hóa để bắt đầu.

def path_removal(current, random_state, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    nr_nodes_to_remove = kwargs.get('nr_nodes_to_remove', determine_nr_nodes_to_remove(current.nb_customers))

    # Tìm một tuyến đường không rỗng một cách ngẫu nhiên
    non_empty_routes = []
    for day, routes in destroyed_solution.schedule.items():
        for r_idx, route_info in enumerate(routes):
            if route_info[2]: # Nếu customer_list không rỗng
                non_empty_routes.append((day, r_idx))
    
    if not non_empty_routes:
        return destroyed_solution, []
    
    day_to_edit, route_to_edit_idx = random_state.choice(non_empty_routes)
    
    route_list = destroyed_solution.schedule[day_to_edit][route_to_edit_idx][2]
    
    seg_length = min(len(route_list), nr_nodes_to_remove)
    if seg_length <= 0:
        return destroyed_solution, []
        
    start = random_state.randint(0, len(route_list) - seg_length)
    customers_to_remove = route_list[start : start + seg_length]
    
    destroyed_solution.schedule = _remove_customers_from_schedule(destroyed_solution.schedule, customers_to_remove)
    return destroyed_solution, customers_to_remove

# Các hàm còn lại có thể được để trống hoặc trả về random_removal để bắt đầu
def relatedness_removal(current, random_state, **kwargs):
    # Tạm thời dùng Shaw removal vì logic tương tự
    return shaw_distance_removal(current, random_state, **kwargs)

def neighbor_graph_removal(current, random_state, **kwargs):
    # Logic này phức tạp, tạm thời dùng random
    return random_removal(current, random_state, **kwargs)

def sequence_removal(current, random_state, **kwargs):
    # Logic tương tự path_removal
    return path_removal(current, random_state, **kwargs)

def history_biased_removal(current, random_state, **kwargs):
     # Logic này phức tạp, tạm thời dùng random
    return random_removal(current, random_state, **kwargs)

def cost_outlier_removal(current, random_state, **kwargs):
    # Logic tương tự worst_removal
    return worst_removal(current, random_state, **kwargs)

def two_route_exchange(current, random_state, **kwargs):
    # Logic này cần được suy nghĩ lại cho scheduling, tạm thời dùng random
    return random_removal(current, random_state, **kwargs)

def depot_shift_removal(current, random_state, **kwargs):
    # Logic này phức tạp, tạm thời dùng random
    return random_removal(current, random_state, **kwargs)

def load_balancing_removal(current, random_state, **kwargs):
     # Logic này phức tạp, tạm thời dùng random
    return random_removal(current, random_state, **kwargs)