import numpy as np
from collections import defaultdict

def compute_initial_solution(problem_instance, random_state, num_days_in_cycle=7):
    # --- LẤY DỮ LIỆU ---
    farms = problem_instance['farms']
    facilities = problem_instance['facilities']
    available_trucks = problem_instance['fleet']['available_trucks']
    # SỬA LẠI TÊN KEY CHO ĐÚNG
    dist_depot_data = problem_instance['distance_depots_farms'] 
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map']

    # --- XÁC ĐỊNH LỊCH GHÉ THĂM ---
    visits_by_day = defaultdict(list)
    for farm in farms:
        farm_id, frequency = farm['id'], farm.get('frequency', 0)
        visit_days = []
        if frequency >= 1: visit_days = range(num_days_in_cycle)
        elif frequency == 0.5: visit_days = range(0, num_days_in_cycle, 2)
        for day in visit_days:
            visits_by_day[day].append(farm_id)

    # --- XÂY DỰNG LỊCH TRÌNH ---
    final_schedule = {day: [] for day in range(num_days_in_cycle)}
    trucks_by_region = defaultdict(list)
    for truck in available_trucks:
        trucks_by_region[truck['region'].strip()].append(truck)

    for day in range(num_days_in_cycle):
        farm_ids_today = visits_by_day.get(day)
        if not farm_ids_today: continue

        customers_by_depot = defaultdict(list)
        for farm_id in farm_ids_today:
            farm_idx = farm_id_to_idx_map.get(farm_id)
            if farm_idx is None: continue
            closest_depot_idx = np.argmin(dist_depot_data[:, farm_idx])
            customers_by_depot[closest_depot_idx].append(farm_id)
            
        routes_for_this_day = []
        for depot_idx, customer_ids in customers_by_depot.items():
            depot_region = facilities[depot_idx]['region'].strip()
            eligible_trucks = trucks_by_region.get(depot_region)
            if not eligible_trucks: continue

            unvisited_customers = list(customer_ids)
            while unvisited_customers:
                selected_truck = random_state.choice(eligible_trucks)
                truck_id, truck_capacity = selected_truck['id'], selected_truck['capacity']
                
                route_customers, route_load = [], 0
                customers_added = []
                for customer_id in unvisited_customers:
                    farm_idx = farm_id_to_idx_map[customer_id]
                    demand = farms[farm_idx]['demand']
                    if route_load + demand <= truck_capacity:
                        route_customers.append(customer_id)
                        route_load += demand
                        customers_added.append(customer_id)
                
                if not customers_added:
                    largest_truck = max(eligible_trucks, key=lambda t: t['capacity'])
                    first_cust_id = unvisited_customers[0]
                    first_cust_demand = farms[farm_id_to_idx_map[first_cust_id]]['demand']
                    if largest_truck['capacity'] >= first_cust_demand:
                        routes_for_this_day.append((depot_idx, largest_truck['id'], [first_cust_id]))
                        unvisited_customers.pop(0)
                        continue
                    else:
                        print(f"CẢNH BÁO: Nông trại {first_cust_id} có demand ({first_cust_demand}) quá lớn. Bỏ qua.")
                        unvisited_customers.pop(0)
                        continue
                
                unvisited_customers = [c for c in unvisited_customers if c not in customers_added]
                if route_customers:
                    routes_for_this_day.append((depot_idx, truck_id, route_customers))
        
        final_schedule[day] = routes_for_this_day
    
    return final_schedule