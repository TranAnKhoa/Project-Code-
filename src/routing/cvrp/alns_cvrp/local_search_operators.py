import copy
from .repair_operators import _calculate_route_schedule_and_feasibility, find_truck_by_id

# ==============================================================================
# HÀM TRỢ GIÚP: TÍNH TOÁN CHI PHÍ CỦA MỘT TUYẾN ĐƯỜNG
# ==============================================================================

def get_route_cost(problem_instance, route_info):
    """
    Tính toán tổng chi phí (di chuyển + chờ) của một tuyến đường duy nhất.
    Sử dụng hàm tính toán đã được nâng cấp để đảm bảo tính nhất quán.
    """
    depot_idx, truck_id, customer_list, shift = route_info
    
    if not customer_list:
        return 0

    truck_info = find_truck_by_id(truck_id, problem_instance['fleet']['available_trucks'])
    if not truck_info:
        return float('inf') # Trả về chi phí vô cùng lớn nếu không tìm thấy xe

    # Lấy các hằng số chi phí
    WAIT_COST_PER_MIN = 0.2
    var_cost_per_km = problem_instance['costs']['variable_cost_per_km'].get(
        (truck_info['type'], truck_info['region']), 1.0
    )

    # Gọi hàm tính toán chính để lấy kết quả chính xác
    _, is_feasible, total_dist, total_wait, _ = _calculate_route_schedule_and_feasibility(
        depot_idx, customer_list, shift, 0, problem_instance, truck_info
    )

    if not is_feasible:
        return float('inf') # Chi phí vô cùng lớn nếu tuyến không khả thi

    # Tổng chi phí của tuyến = chi phí di chuyển + chi phí chờ
    return (total_dist * var_cost_per_km) + (total_wait * WAIT_COST_PER_MIN)


# ==============================================================================
# TOÁN TỬ LOCAL SEARCH: 2-OPT
# ==============================================================================

def apply_2_opt(solution):
    """
    Áp dụng thuật toán 2-Opt để tối ưu hóa thứ tự các điểm dừng bên trong mỗi tuyến.
    """
    improved_solution = copy.deepcopy(solution)
    problem_instance = improved_solution.problem_instance
    
    # Lặp qua từng ngày trong lịch trình
    for day in improved_solution.schedule:
        routes_on_day = improved_solution.schedule[day]
        
        # Lặp qua từng tuyến đường trong ngày
        for route_idx, route_info in enumerate(routes_on_day):
            depot_idx, truck_id, customer_list, shift = route_info
            
            # 2-Opt chỉ có ý nghĩa với các tuyến có ít nhất 2 điểm dừng
            if len(customer_list) < 2 or shift == 'INTER-FACTORY':
                continue

            improved_in_route = True
            while improved_in_route:
                improved_in_route = False
                
                # Tính chi phí tốt nhất hiện tại của tuyến
                best_cost = get_route_cost(problem_instance, (depot_idx, truck_id, customer_list, shift))

                # Lặp qua tất cả các cặp cạnh (i, j) để thử "đảo ngược"
                # i từ 0 đến n-2
                for i in range(len(customer_list) - 1):
                    # j từ i+1 đến n-1
                    for j in range(i + 1, len(customer_list)):
                        # Tạo một tuyến đường mới bằng cách đảo ngược đoạn giữa i và j
                        # Ví dụ: [A, B, C, D, E], i=1 (B), j=3 (D)
                        # new_list = [A] + [D, C, B] + [E]
                        new_customer_list = (
                            customer_list[:i+1] + 
                            customer_list[i+1:j+1][::-1] + 
                            customer_list[j+1:]
                        )
                        
                        # Tạo route_info mới để kiểm tra
                        new_route_info = (depot_idx, truck_id, new_customer_list, shift)
                        
                        # Tính chi phí của tuyến đường mới
                        new_cost = get_route_cost(problem_instance, new_route_info)
                        
                        # Nếu tìm thấy một sự cải thiện
                        if new_cost < best_cost:
                            # Cập nhật lại customer_list, chi phí tốt nhất và đánh dấu đã có cải thiện
                            customer_list = new_customer_list
                            best_cost = new_cost
                            improved_in_route = True
                            
                            # Thoát khỏi vòng lặp j và i để bắt đầu lại từ đầu với tuyến đã được cải thiện
                            # (Đây là chiến lược "first improvement")
                            break 
                    if improved_in_route:
                        break
            
            # Sau khi tối ưu hóa xong, cập nhật lại tuyến đường trong lịch trình
            improved_solution.schedule[day][route_idx] = (depot_idx, truck_id, customer_list, shift)
            
    return improved_solution