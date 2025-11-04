import numpy as np
import re

# ==============================================================================
# HÀM TIỆN ÍCH (Giữ nguyên như trước)
# ==============================================================================

def find_truck_by_id(truck_id, available_trucks):
    """Tiện ích để tìm thông tin chi tiết của xe từ ID."""
    for truck in available_trucks:
        if truck['id'] == truck_id:
            return truck
    return None

def _clean_base_id(fid):
    """Làm sạch ID để lấy ID gốc của nông trại vật lý."""
    if not isinstance(fid, str):
        return fid
    return re.split(r'(_onfly.*|_fallback_part.*|_part.*|_d\d+)', fid)[0]

# ==============================================================================
# ĐỊNH NGHĨA CLASS cvrpEnv
# ==============================================================================

class cvrpEnv:
    def __init__(self, initial_schedule, problem_instance, seed, **kwargs):
        self.problem_instance = problem_instance
        self.schedule = initial_schedule
        self.seed = seed
        self.dist_matrix_data = problem_instance['distance_matrix_farms']
        self.dist_depot_data = problem_instance['distance_depots_farms']
        self.farm_id_to_idx = problem_instance['farm_id_to_idx_map']
        self.num_facilities = len(problem_instance['facilities'])
        self.num_farms = len(problem_instance['farms'])
        self.nb_customers = self.num_farms
        self.demands_data = [farm['demand'] for farm in problem_instance['farms']]
        self.customer_tw = [[farm['time_windows']['AM'][0], farm['time_windows']['PM'][1]] for farm in problem_instance['farms']]
        self.customer_st = [farm['service_time_params'][0] for farm in problem_instance['farms']]
        self.depot_tw = [[0, 24*60] for _ in problem_instance['facilities']]
        self.truck_capacity = problem_instance['fleet']['available_trucks'][0]['capacity'] if problem_instance['fleet']['available_trucks'] else 0

    def _get_farm_idx(self, farm_id):
        """Hàm tra cứu ID nông trại một cách "bền bỉ"."""
        try:
            return self.farm_id_to_idx[farm_id]
        except KeyError:
            try:
                return self.farm_id_to_idx[int(farm_id)]
            except (KeyError, ValueError):
                # (Phần in thông tin debug giữ nguyên)
                raise KeyError(f"Không thể tìm thấy Farm ID '{farm_id}' trong farm_id_to_idx map.")

    def objective(self):
        """
        Tính toán objective function, bao gồm chi phí vận hành, chi phí cố định,
        chi phí chờ đợi và phạt vi phạm thời gian.
        *** PHIÊN BẢN HOÀN CHỈNH ĐÃ TÍCH HỢP VELOCITY VÀ WAITING COST ***
        """
        # --- CÁC HẰNG SỐ CHI PHÍ ---
        PENALTY_FACTOR = 1000 
        WAIT_COST_PER_MIN = 0.2 # Chi phí cho mỗi phút chờ

        # --- KHỞI TẠO CÁC BIẾN CHI PHÍ ---
        total_variable_cost = 0.0
        total_fixed_cost = 0.0
        total_time_window_penalty = 0.0
        total_waiting_cost = 0.0
        
        unique_trucks_used = set()
        num_days_in_cycle = len(self.schedule) if self.schedule else 1
        virtual_map = self.problem_instance.get('virtual_split_farms', {})
        dist_depot_depot = self.problem_instance.get('distance_matrix_depots', None)

        if not self.schedule:
            return 0.0, 0.0

        for day in self.schedule:
            for route_info in self.schedule[day]:
                depot_idx, truck_id, customer_list, shift = route_info
                if not customer_list and shift != 'INTER-FACTORY': continue

                unique_trucks_used.add(truck_id)
                truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
                
                # Xác định var_cost và velocity cho từng tuyến
                var_cost_per_km = 1.0
                velocity = 1.0
                if truck_details:
                    truck_type, truck_region = truck_details['type'], truck_details['region']
                    var_cost_per_km = self.problem_instance['costs']['variable_cost_per_km'].get((truck_type, truck_region), 1.0)
                    velocity = 1.0 if truck_type in ["Single", "Truck and Dog"] else 0.5
                
                if shift == 'INTER-FACTORY':
                    if dist_depot_depot is not None:
                        try:
                            parts = customer_list[0].split('_')
                            from_depot = int(parts[2])
                            to_depot = int(parts[4])
                            travel_dist = dist_depot_depot[from_depot, to_depot] * 2
                            total_variable_cost += travel_dist * var_cost_per_km
                        except (IndexError, ValueError): pass
                    continue

                current_time = self.depot_tw[depot_idx][0]
                depot_end_time = self.depot_tw[depot_idx][1]

                # --- 1. Từ Depot đến farm đầu tiên ---
                first_customer_id = customer_list[0]
                first_idx = self._get_farm_idx(_clean_base_id(first_customer_id))
                
                travel_dist = self.dist_depot_data[depot_idx, first_idx]
                travel_time = travel_dist / velocity
                
                total_variable_cost += travel_dist * var_cost_per_km
                arrival_time = current_time + travel_time

                farm_info = self.problem_instance['farms'][first_idx]
                start_tw, end_tw = farm_info['time_windows'][shift]
                fix_time, var_param = farm_info['service_time_params']
                demand = virtual_map.get(first_customer_id, {}).get('portion', farm_info['demand'])
                service_time = fix_time + (demand / var_param if var_param > 0 else 0)

                if arrival_time < start_tw:
                    total_waiting_cost += (start_tw - arrival_time) * WAIT_COST_PER_MIN
                    service_start = start_tw
                else:
                    service_start = arrival_time
                
                if service_start > end_tw: # Đến hoặc bắt đầu phục vụ quá trễ
                    total_time_window_penalty += (service_start - end_tw) * PENALTY_FACTOR

                current_time = service_start + service_time

                # --- 2. Giữa các farm ---
                for i in range(len(customer_list) - 1):
                    from_idx = self._get_farm_idx(_clean_base_id(customer_list[i]))
                    to_idx = self._get_farm_idx(_clean_base_id(customer_list[i+1]))

                    travel_dist = self.dist_matrix_data[from_idx, to_idx]
                    travel_time = travel_dist / velocity
                    
                    total_variable_cost += travel_dist * var_cost_per_km
                    arrival_time = current_time + travel_time

                    farm_info = self.problem_instance['farms'][to_idx]
                    start_tw, end_tw = farm_info['time_windows'][shift]
                    fix_time, var_param = farm_info['service_time_params']
                    demand = virtual_map.get(customer_list[i+1], {}).get('portion', farm_info['demand'])
                    service_time = fix_time + (demand / var_param if var_param > 0 else 0)

                    if arrival_time < start_tw:
                        total_waiting_cost += (start_tw - arrival_time) * WAIT_COST_PER_MIN
                        service_start = start_tw
                    else:
                        service_start = arrival_time

                    if service_start > end_tw:
                        total_time_window_penalty += (service_start - end_tw) * PENALTY_FACTOR

                    current_time = service_start + service_time

                # --- 3. Quay về depot ---
                last_idx = self._get_farm_idx(_clean_base_id(customer_list[-1]))
                travel_dist_back = self.dist_depot_data[depot_idx, last_idx]
                travel_time_back = travel_dist_back / velocity
                
                total_variable_cost += travel_dist_back * var_cost_per_km
                arrival_back = current_time + travel_time_back
                
                if arrival_back > depot_end_time:
                    total_time_window_penalty += (arrival_back - depot_end_time) * PENALTY_FACTOR

        # --- Chi phí thuê xe ---
        for truck_id in unique_trucks_used:
            truck_details = find_truck_by_id(truck_id, self.problem_instance['fleet']['available_trucks'])
            if truck_details:
                lease_cost_per_day = truck_details.get('lease_cost_monthly', 0) / 30
                total_fixed_cost += lease_cost_per_day * num_days_in_cycle

        # Tổng hợp tất cả các thành phần chi phí
        total_cost = total_variable_cost + total_fixed_cost + total_time_window_penalty + total_waiting_cost
        
        # Trả về tổng chi phí và tổng phạt (để có thể theo dõi riêng)
        return total_cost, total_time_window_penalty