import numpy as np

# Đặt hàm này bên ngoài class để có thể sử dụng ở nhiều nơi
def find_truck_by_id(truck_id, available_trucks):
    """Tiện ích để tìm thông tin chi tiết của xe từ ID."""
    for truck in available_trucks:
        if truck['id'] == truck_id:
            return truck
    return None

class cvrpEnv:
    def __init__(self, initial_schedule, problem_instance, seed, **kwargs):
        self.problem_instance = problem_instance
        self.schedule = initial_schedule
        self.seed = seed

        # SỬA LẠI CÁC TRUY CẬP KEY CHO ĐÚNG
        self.dist_matrix_data = problem_instance['distance_matrix_farms']
        self.dist_depot_data = problem_instance['distance_depots_farms']
        self.farm_id_to_idx = problem_instance['farm_id_to_idx_map']

        # Gán các thuộc tính cũ để tương thích
        self.num_facilities = len(problem_instance['facilities'])
        self.num_farms = len(problem_instance['farms'])
        self.nb_customers = self.num_farms
        self.demands_data = [farm['demand'] for farm in problem_instance['farms']]
        self.customer_tw = [[farm['time_windows']['AM'][0], farm['time_windows']['PM'][1]] for farm in problem_instance['farms']]
        self.customer_st = [farm['service_time_params'][0] for farm in problem_instance['farms']]
        self.depot_tw = [[0, 24*60] for _ in problem_instance['facilities']]
        self.truck_capacity = problem_instance['fleet']['available_trucks'][0]['capacity'] if problem_instance['fleet']['available_trucks'] else 0

    def objective(self):
        """
        Tính toán objective function cho toàn bộ LỊCH TRÌNH (self.schedule).
        Objective = Tổng chi phí (biến đổi + cố định) + Tổng hình phạt.
        """
        PENALTY_FACTOR = 1000 

        total_variable_cost = 0.0
        total_fixed_cost = 0.0
        total_time_window_penalty = 0.0
        
        unique_trucks_used = set()
        num_days_in_cycle = len(self.schedule) if self.schedule else 1

        if not self.schedule:
            return 0.0, 0.0

        # <<< VÒNG LẶP THEO NGÀY >>>
        for day in self.schedule:
            routes_of_day = self.schedule[day]

            for route_info in routes_of_day:
                depot_idx, truck_id, customer_id_list = route_info
                
                if not customer_id_list:
                    continue

                unique_trucks_used.add(truck_id)
                
                truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
                var_cost_per_km = 1.0 # Giá trị mặc định
                if truck_details:
                    truck_type = truck_details['type']
                    truck_region = truck_details['region']
                    var_cost_per_km = self.problem_instance['costs']['variable_cost_per_km'].get((truck_type, truck_region), 1.0)

                depot_start_time = self.depot_tw[depot_idx][0]
                depot_end_time = self.depot_tw[depot_idx][1]
                current_time = depot_start_time
                
                # --- SỬA LỖI INDEX TẠI ĐÂY ---
                
                # 1. Từ Depot đến khách hàng đầu tiên
                first_customer_id = customer_id_list[0]
                first_customer_idx = self.farm_id_to_idx[first_customer_id] # DÙNG MAP ÁNH XẠ

                travel_time_to_first = self.dist_depot_data[depot_idx, first_customer_idx]
                total_variable_cost += travel_time_to_first * var_cost_per_km
                arrival_time = current_time + travel_time_to_first

                cust_start_tw, cust_end_tw = self.customer_tw[first_customer_idx]
                cust_service_time = self.customer_st[first_customer_idx]

                if arrival_time > cust_end_tw:
                    total_time_window_penalty += (arrival_time - cust_end_tw) * PENALTY_FACTOR
                
                service_start_time = max(arrival_time, cust_start_tw)
                current_time = service_start_time + cust_service_time

                # 2. Giữa các khách hàng
                for i in range(len(customer_id_list) - 1):
                    from_customer_id = customer_id_list[i]
                    to_customer_id = customer_id_list[i+1]
                    
                    from_idx = self.farm_id_to_idx[from_customer_id] # DÙNG MAP ÁNH XẠ
                    to_idx = self.farm_id_to_idx[to_customer_id]     # DÙNG MAP ÁNH XẠ
                    
                    travel_time_between = self.dist_matrix_data[from_idx, to_idx]
                    total_variable_cost += travel_time_between * var_cost_per_km
                    arrival_time = current_time + travel_time_between

                    next_cust_start_tw, next_cust_end_tw = self.customer_tw[to_idx]
                    next_cust_service_time = self.customer_st[to_idx]

                    if arrival_time > next_cust_end_tw:
                        total_time_window_penalty += (arrival_time - next_cust_end_tw) * PENALTY_FACTOR

                    service_start_time = max(arrival_time, next_cust_start_tw)
                    current_time = service_start_time + next_cust_service_time

                # 3. Quay về Depot
                last_customer_id = customer_id_list[-1]
                last_customer_idx = self.farm_id_to_idx[last_customer_id] # DÙNG MAP ÁNH XẠ

                travel_time_to_depot = self.dist_depot_data[depot_idx, last_customer_idx]
                total_variable_cost += travel_time_to_depot * var_cost_per_km
                arrival_at_depot = current_time + travel_time_to_depot
                
                if arrival_at_depot > depot_end_time:
                    total_time_window_penalty += (arrival_at_depot - depot_end_time) * PENALTY_FACTOR
        
        # --- TÍNH CHI PHÍ CỐ ĐỊNH (Tạm thời chỉ tính chi phí thuê) ---
        for truck_id in unique_trucks_used:
            truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
            if truck_details:
                lease_cost_per_day = truck_details.get('lease_cost_monthly', 0) / 30
                total_fixed_cost += lease_cost_per_day * num_days_in_cycle

        final_objective = total_variable_cost + total_fixed_cost + total_time_window_penalty
        return final_objective, total_time_window_penalty