import numpy as np
from collections import defaultdict
import random
import re
import copy


# ======================= HÀM TIỆN ÍCH =======================
def _clean_base_id(fid):
    """Remove suffixes like _onfly, _part, _d<number> to get the real farm id."""
    if not isinstance(fid, str):  # Nếu fid không phải chuỗi (VD: None, số...), trả nguyên giá trị
        return fid
    # Regex để tách bỏ các hậu tố farm ảo (_onfly, _part, _fallback_part, _d1, _d2,...)
    return re.split(r'(_onfly.*|_fallback_part.*|_part.*|_d\d+)', fid)[0]  



def _resolve_farm_for_ci(fid, problem_instance, farms, farm_id_to_idx_map):
    """
    Safely resolve farm id (real or virtual). Supports multi-layer virtual ids.
    Returns (base_id, portion_demand, farm_info, farm_idx)
    → Dùng để "giải mã" farm ảo về farm thật, kể cả khi farm ảo được chia nhiều tầng.
    """
    virtual_map = problem_instance.setdefault('virtual_split_farms', {})  # Map chứa các farm ảo (nếu chưa có thì tạo dict rỗng)

    # ===== CASE 1: farm_id là farm ảo thật sự (đã có trong virtual_map)
    if isinstance(fid, str) and fid in virtual_map:
        base = virtual_map[fid]['base_id']          # Lấy id gốc (base farm thật)
        portion = virtual_map[fid].get('portion', 0) # Lấy phần demand (số lượng sữa phần ảo này cần)
        visited = set()                             # Dùng để phát hiện vòng lặp nếu farm ảo tham chiếu chính nó

        # Vòng lặp: tìm đến farm thật cuối cùng
        while base not in farm_id_to_idx_map:
            if base in visited:
                raise KeyError(f"Cycle detected in virtual_split_farms for '{fid}', base '{base}'.")  # Nếu loop vô hạn
            visited.add(base)
            if base in virtual_map:
                base = virtual_map[base]['base_id']  # Nếu base lại là ảo → lặp tiếp
            else:
                base = _clean_base_id(base)          # Nếu base chỉ là dạng chuỗi có "_part..." → làm sạch
                if base in farm_id_to_idx_map:
                    break  # Khi đã tìm thấy farm thật
                raise KeyError(f"⚠️ Base farm '{base}' for virtual id '{fid}' not found in farm_id_to_idx_map.")
        
        # <<< SỬA LỖI TRA CỨU AN TOÀN >>>
        try:
            base_idx = farm_id_to_idx_map[base]
        except KeyError:
            base_idx = farm_id_to_idx_map[int(base)]

        base_info = farms[base_idx]                  # Lấy toàn bộ thông tin của farm thật
        return base, portion, base_info, base_idx    # Trả lại tuple 4 giá trị

    # ===== CASE 2: fid trông giống farm ảo (chứa "_part", "_onfly"...) nhưng chưa có mapping
    if isinstance(fid, str) and ("_part" in fid or "_onfly" in fid or "_fallback_part" in fid or re.search(r'_d\d+', fid)):
        base = _clean_base_id(fid)                   # Làm sạch hậu tố để lấy base_id
        # <<< SỬA LỖI TRA CỨU AN TOÀN >>>
        try:
            base_idx = farm_id_to_idx_map[base]
        except KeyError:
            base_idx = farm_id_to_idx_map[int(base)]
            
        base_info = farms[base_idx]
        return base, 0, base_info, base_idx      # portion=0 vì không có lượng chia rõ ràng

    # ===== CASE 3: fid là farm thật
    # <<< SỬA LỖI TRA CỨU AN TOÀN >>>
    try:
        idx = farm_id_to_idx_map[fid]
    except KeyError:
        try:
            idx = farm_id_to_idx_map[int(fid)]
        except (KeyError, ValueError):
             raise KeyError(f"Farm id '{fid}' not present in farm_id_to_idx_map.")

    info = farms[idx]
    return fid, info['demand'], info, idx            # Trả lại id, demand, info, và index


def _calculate_route_schedule_and_feasibility(depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info):
    """
    Kiểm tra tính khả thi của route với time window và capacity.
    Trả về: (finish_time_at_depot, feasible)
    """
    if not customer_list:            # Nếu route rỗng → không có khách hàng
        return start_time_at_depot, True

    dist_matrix = problem_instance['distance_matrix_farms']       # Ma trận khoảng cách giữa các farm
    depot_farm_dist = problem_instance['distance_depots_farms']   # Ma trận khoảng cách giữa depot và farm
    farms = problem_instance['farms']                             # Danh sách thông tin farm
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']       # Map farm_id → index
    depot_end_time = 1440                                         # Thời điểm kết thúc ngày (tính bằng phút)
    current_time = start_time_at_depot                            # Thời gian hiện tại (bắt đầu từ depot)
    truck_name = truck_info['type']                               # Loại xe (Single, B-Double, v.v.)
    velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5  # Tốc độ di chuyển (xe to chạy nhanh hơn)
    virtual_map = problem_instance.get('virtual_split_farms', {}) # Lấy mapping farm ảo (nếu có)

    # --- Hàm con để xử lý farm thật/ảo trong danh sách ---
    def _resolve_farm(fid):
        """Trả về thông tin farm thật (có thể là từ farm ảo)."""
        base_id_str = _clean_base_id(fid)
        try:
            base_idx = farm_id_to_idx[base_id_str]
        except KeyError:
            base_idx = farm_id_to_idx[int(base_id_str)]
        
        base_info = farms[base_idx]

        if isinstance(fid, str) and fid in virtual_map:
            portion = virtual_map[fid]['portion']
            return base_idx, portion, base_info['service_time_params'], base_info['time_windows']
        else:
            return base_idx, base_info['demand'], base_info['service_time_params'], base_info['time_windows']

    # ======== Bắt đầu kiểm tra route ========

    # 1️⃣ depot → farm đầu tiên
    first_cust_id = customer_list[0]                              # ID farm đầu tiên
    first_idx, first_demand, first_params, first_tw = _resolve_farm(first_cust_id)
    travel_time = depot_farm_dist[depot_idx, first_idx] / velocity # Thời gian di chuyển
    arrival_time = current_time + travel_time                      # Thời điểm tới nơi

    start_tw, end_tw = first_tw[shift]                             # Lấy time window (theo ca: AM hoặc PM)
    if arrival_time > end_tw:                                      # Nếu đến trễ hơn cửa sổ thời gian cho phép
        return -1, False                                           # Route không khả thi

    service_start = max(arrival_time, start_tw)                    # Bắt đầu phục vụ tại farm
    fix_time, var_param = first_params                             # Thời gian cố định + biến thiên theo demand
    service_duration = fix_time + (first_demand / var_param if var_param > 0 else 0)
    current_time = service_start + service_duration                # Cập nhật thời gian sau khi phục vụ xong

    # 2️⃣ farm → farm tiếp theo
    for i in range(len(customer_list) - 1):
        from_idx, _, _, _ = _resolve_farm(customer_list[i])        # Farm hiện tại
        to_idx, to_demand, to_params, to_tw = _resolve_farm(customer_list[i + 1]) # Farm kế tiếp
        travel_time = dist_matrix[from_idx, to_idx] / velocity     # Thời gian di chuyển giữa 2 farm
        arrival_time = current_time + travel_time
        start_tw, end_tw = to_tw[shift]
        if arrival_time > end_tw:                                  # Nếu tới trễ hơn time window
            return -1, False
        service_start = max(arrival_time, start_tw)
        fix_time, var_param = to_params
        service_duration = fix_time + (to_demand / var_param if var_param > 0 else 0)
        current_time = service_start + service_duration            # Cập nhật thời gian hiện tại

    # 3️⃣ farm cuối cùng → quay về depot
    last_idx, _, _, _ = _resolve_farm(customer_list[-1])           # Farm cuối
    travel_time_back = depot_farm_dist[depot_idx, last_idx] / velocity  # Thời gian quay lại depot
    finish_time_at_depot = current_time + travel_time_back         # Tổng thời gian kết thúc
    if finish_time_at_depot > depot_end_time:                      # Nếu vượt quá giới hạn ngày
        return -1, False                                           # Không khả thi

    return finish_time_at_depot, True                              # Trả kết quả thành công

# ==================== HÀM CHÍNH ====================
def compute_initial_solution(problem_instance, random_state, num_days_in_cycle=7):
    """
    Sinh lời giải ban đầu cho bài toán VRP phức tạp.
    Bao gồm: multi-depot, time windows, multi-trip, split demand (on-the-fly), và route expansion.
    """
    print("\n--- BÊN TRONG COMPUTE_INITIAL_SOLUTION (ĐÃ NÂNG CẤP ROUTE EXPANSION) ---")  # In thông báo bắt đầu hàm
    count = 0                      # Biến đếm số farm không thể lên lịch (sẽ in ra cuối)
    split_done = set()             # Set để theo dõi các farm đã split (nếu dùng logic split khác)
    onfly_split_done = set()       # Set để tránh split on-the-fly nhiều lần cho cùng (day, farm)

    farms = problem_instance['farms']                                # Danh sách dict thông tin từng farm
    facilities = problem_instance['facilities']                      # Danh sách depot / facility
    available_trucks = problem_instance['fleet']['available_trucks'] # Danh sách xe có sẵn (list of dict)
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map']      # Map từ farm_id → index trong farms list
    final_schedule = {day: [] for day in range(num_days_in_cycle)}   # Khởi tạo lịch rỗng cho mỗi ngày trong chu kỳ
    
    ## LOGIC MỚI: KHỞI TẠO CÁC BIẾN CHO INTER-FACTORY TRANSFER ##
    # Giả định một sức chứa mặc định cho mỗi kho. Bạn có thể thay đổi giá trị này.
    DEPOT_CAPACITY = 50000 
    # Biến cục bộ để theo dõi lượng sữa đã thu về mỗi kho trong mỗi ngày
    depot_daily_load = {day: defaultdict(float) for day in range(num_days_in_cycle)}
    # Gom nhóm các kho theo vùng để dễ tìm kho đích để chuyển sữa đến
    depots_by_region = defaultdict(list)
    for i, facility in enumerate(facilities):
        if 'region' in facility:
            depots_by_region[facility['region']].append(i)
    ## KẾT THÚC LOGIC MỚI ##

    # tạo danh sách farm cần phục vụ (day, farm_id) dựa trên frequency
    all_required_visits = []  # sẽ chứa tuples (day_idx, farm_id)
    for farm in farms:
        farm_id, frequency = farm['id'], farm.get('frequency', 0)  # Lấy id và tần suất phục vụ (ví dụ 1, 0.5,...)
        # Nếu frequency >=1 => phục vụ mỗi ngày trong chu kỳ
        # Nếu frequency == 0.5 => phục vụ cách ngày (0,2,4,...)
        # Nếu frequency == 0 => không phục vụ
        if frequency >= 1:
            visit_days = range(num_days_in_cycle)                     # mỗi ngày
        elif frequency == 0.5:
            visit_days = range(0, num_days_in_cycle, 2)               # cách ngày
        else:
            visit_days = []                                          # không cần phục vụ
        for d in visit_days:
            all_required_visits.append((d, farm_id))                 # thêm tuple (ngày, farm_id) vào danh sách

    random_state.shuffle(all_required_visits)  # Trộn ngẫu nhiên thứ tự để đa dạng lời giải ban đầu
    # truck_finish_times lưu thời gian kết thúc cuối cùng của mỗi truck mỗi ngày (để cho phép multi-trip)
    truck_finish_times = {day: defaultdict(lambda: (0, -1)) for day in range(num_days_in_cycle)}
    assigned_farms = set()                      # tập các farm đã được gán (day, farm_id)
    virtual_map = problem_instance.setdefault('virtual_split_farms', {})  # đảm bảo có dict cho farm ảo

    # Hàm nội bộ riêng cho compute để xử lý farm ảo nhiều tầng (giống _resolve_farm_for_ci nhưng dùng closure)
    def _resolve_farm_for_ci_local(fid):
        """Xử lý mapping farm ảo nhiều tầng về farm thật, trả (base, portion, base_info, base_idx)."""
        # Đây là phiên bản an toàn hơn của _resolve_farm_for_ci, sử dụng closure để truy cập biến bên ngoài
        if isinstance(fid, str) and fid in virtual_map:
            base = virtual_map[fid]['base_id']
            portion = virtual_map[fid].get('portion', 0)
            visited = set()
            while True:
                is_base_in_map = False
                try: # Thử tra cứu bằng key gốc (có thể là str hoặc int)
                    if base in farm_id_to_idx_map: is_base_in_map = True
                except TypeError: # Nếu key là int và base là str có thể gây lỗi, hoặc ngược lại
                    pass

                if is_base_in_map: break

                if base in visited: raise KeyError(f"Cycle in virtual_split_farms for '{fid}', base '{base}'.")
                visited.add(base)
                if base in virtual_map:
                    base = virtual_map[base]['base_id']
                else:
                    base = _clean_base_id(base)
                    try: # Thử lại tra cứu sau khi làm sạch
                        if base in farm_id_to_idx_map: break
                        if int(base) in farm_id_to_idx_map: base = int(base); break
                    except (ValueError, TypeError): pass
                    raise KeyError(f"⚠️ Base farm '{base}' for virtual id '{fid}' not found in farm_id_to_idx_map.")
            
            try: base_idx = farm_id_to_idx_map[base]
            except KeyError: base_idx = farm_id_to_idx_map[int(base)]
            
            base_info = farms[base_idx]
            return base, portion, base_info, base_idx

        base_clean = _clean_base_id(fid)
        try: idx = farm_id_to_idx_map[base_clean]
        except KeyError:
            try: idx = farm_id_to_idx_map[int(base_clean)]
            except (KeyError, ValueError): raise KeyError(f"Farm id '{fid}' not present in farm_id_to_idx_map.")
        
        info = farms[idx]
        return fid, info['demand'], info, idx

    # ====================== MAIN LOOP ======================
    i = 0
    while i < len(all_required_visits):               # duyệt toàn bộ danh sách farm cần phục vụ
        day_idx, farm_id_to_insert = all_required_visits[i]  # lấy tuple (ngày, farm)
        i += 1
        if (day_idx, farm_id_to_insert) in assigned_farms:   # nếu đã gán rồi thì bỏ qua
            continue

        # Resolve farm (có thể trả về farm gốc và phần demand nếu là farm ảo)
        effective_id, eff_demand, farm_details, farm_idx = _resolve_farm_for_ci_local(farm_id_to_insert)

        # Tìm depot gần nhất (min distance từ depot -> farm)
        closest_depot_idx = int(np.argmin(problem_instance['distance_depots_farms'][:, farm_idx]))
        depot_region = facilities[closest_depot_idx].get('region', None)  # Lấy region của depot gần nhất

        # Chỉ dùng xe thuộc region đó (nếu xe có thuộc tính region)
        farm_access = farm_details.get('accessibility', None)
# === ACCESSIBILITY CHECK: ánh xạ loại xe sang chỉ số ===
        type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
        farm_access = farm_details.get('accessibility', None)

        eligible_trucks_in_region = []
        for t in available_trucks:
            if t.get('region') != depot_region:
                continue

            # ánh xạ type → type_idx
            t_idx = type_to_idx.get(t.get('type'), 0)
            t['type_idx'] = t_idx  # thêm field để các phần sau dùng được

            if farm_access is None or (len(farm_access) > t_idx and farm_access[t_idx] == 1):
                eligible_trucks_in_region.append(t)


        if not eligible_trucks_in_region:
            # Nếu không có xe trong vùng tương ứng → in cảnh báo và tăng biến count (không thể lên lịch)
            print(f"!!! KHÔNG CÓ XE Ở VÙNG {depot_region} để phục vụ Farm {farm_id_to_insert}")
            count += 1
            continue

        # Tìm sức chứa lớn nhất trong region để quyết định split on-the-fly
        max_capacity_in_region = max(t['capacity'] for t in eligible_trucks_in_region)

        # --- ON-THE-FLY SPLIT ---
        # Nếu demand lớn hơn toàn bộ sức chứa xe lớn nhất trong region thì ta tạo các farm ảo (phân phần)
        if eff_demand > max_capacity_in_region and (day_idx, farm_id_to_insert) not in onfly_split_done:
            num_parts = int(np.ceil(eff_demand / max_capacity_in_region))  # số phần cần chia
            remaining = eff_demand                                          # còn lại bao nhiêu
            true_base = _clean_base_id(effective_id)                        # base id thật (loại bỏ hậu tố)
            print(f"⚠️ ON-THE-FLY SPLIT: {farm_id_to_insert} (day {day_idx}) demand {eff_demand} > {max_capacity_in_region}. "
                  f"Tạo {num_parts} phần cho nhiều truck.")
            for k in range(num_parts):
                # phần cho mỗi split: nếu là phần cuối thì lấy toàn bộ remaining, else lấy max_capacity
                part_qty = min(max_capacity_in_region, remaining) if k < num_parts - 1 else remaining
                split_id = f"{farm_id_to_insert}_onfly_part{k+1}_d{day_idx}"  # đặt id farm ảo rõ ràng để truy xuất sau
                virtual_map[split_id] = {'base_id': true_base, 'portion': part_qty}  # lưu vào virtual_map
                all_required_visits.append((day_idx, split_id))  # thêm visit cho phần ảo này vào danh sách xử lý
                remaining -= part_qty
                print(f"   ↳ [Split created] {split_id} → base {true_base}, qty {part_qty}")
            # Đánh dấu farm gốc đã được split để không lặp lại split
            assigned_farms.add((day_idx, farm_id_to_insert))
            onfly_split_done.add((day_idx, farm_id_to_insert))
            continue  # quay lại vòng while để xử lý các phần ảo mới thêm vào


        # --- NORMAL SCHEDULING (KHÔNG CẦN SPLIT) ---
        suitable_trucks = [t for t in eligible_trucks_in_region if t['capacity'] >= eff_demand]
        # suitable_trucks: các xe ở region đó có capacity >= demand (nếu none thì lỗi tải)
        if not suitable_trucks:
            # Nếu không có xe nào đủ tải để phục vụ farm (và farm không được split) → báo lỗi
            print(f"!!! LỖI TẢI TRỌNG: Không có xe đủ tải cho Farm {farm_id_to_insert} ở vùng {depot_region}.")
            count += 1
            continue

        # Tìm phương án tốt nhất: (finish_time nhỏ nhất, option_info)
        best_new_route_option = (float('inf'), None)
        for truck_obj in suitable_trucks:                      # thử từng xe khả dĩ
            truck_id = truck_obj['id']                         # id của xe
            last_finish_time, _ = truck_finish_times[day_idx].get(truck_id, (0, -1))
            # Nếu xe đã có chuyến trước đó trong cùng ngày → bắt đầu sau last_finish_time + 30 phút (turnaround)
            start_time = last_finish_time + 30 if last_finish_time > 0 else 0
            for shift in ['AM', 'PM']:                         # thử cả 2 ca (AM/PM)
                finish_time, feasible = _calculate_route_schedule_and_feasibility(
                    closest_depot_idx, [farm_id_to_insert], shift, start_time, problem_instance, truck_obj
                )  # check route chỉ với farm đơn lẻ
                if feasible and finish_time < best_new_route_option[0]:
                    # Nếu khả thi và finish sớm hơn → cập nhật phương án tốt nhất
                    best_new_route_option = (
                        finish_time,
                        (closest_depot_idx, truck_id, [farm_id_to_insert], shift, finish_time, truck_obj),
                    )

        # Nếu tìm được phương án khả thi
        if best_new_route_option[1] is not None:
            _, (depot, truck, cust_list, chosen_shift, new_finish_time, truck_obj) = best_new_route_option
            # đánh dấu các farm trong cust_list đã được gán (ở đây cust_list ban đầu chỉ có farm hiện tại)
            for fid in cust_list:
                assigned_farms.add((day_idx, fid))

            # residual_capacity: còn bao nhiêu tải sau khi chở cust_list hiện tại
            residual_capacity = truck_obj['capacity'] - sum(
                (_resolve_farm_for_ci_local(fid)[1]) for fid in cust_list
            )

            # Tập candidate_farms: farm khác cùng region, chưa được phục vụ trong ngày, và không phải farm đang có trong cust_list
            candidate_farms = {
                f['id']
                for f in farms
                if f.get('region') == depot_region
                and (day_idx, f['id']) not in assigned_farms
                and f['id'] not in cust_list
            }

            # <<< THAY ĐỔI LỚN: NÂNG CẤP LOGIC MỞ RỘNG TUYẾN ĐƯỜNG >>>
            improved = True
            while improved:
                improved = False
                best_insertion_info = None # Sẽ lưu (farm_id, vị trí chèn)
                best_overall_finish = float('inf')

                # Duyệt từng ứng viên để tìm farm và vị trí chèn tốt nhất
                for fid in list(candidate_farms):
                    _, cand_demand, cand_info, _ = _resolve_farm_for_ci_local(fid)
                    cand_access = cand_info.get('accessibility', None)
                    if cand_access is not None and (len(cand_access) <= truck_obj['type_idx'] or cand_access[truck_obj['type_idx']] == 0):
                        candidate_farms.discard(fid)
                        continue

                    if cand_demand > residual_capacity:
                        candidate_farms.discard(fid)
                        continue
                    
                    # Thử chèn farm 'fid' vào mọi vị trí có thể trên tuyến đường hiện tại
                    for insert_pos in range(len(cust_list) + 1):
                        test_route = cust_list[:insert_pos] + [fid] + cust_list[insert_pos:]
                        finish_time, feasible = _calculate_route_schedule_and_feasibility(
                            depot, test_route, chosen_shift, start_time, problem_instance, truck_obj
                        )
                        
                        # Nếu tìm được một vị trí chèn khả thi và tốt hơn
                        if feasible and finish_time < best_overall_finish:
                            best_overall_finish = finish_time
                            best_insertion_info = (fid, insert_pos)
                
                # Nếu đã tìm được một cách chèn tốt nhất trong vòng lặp trên
                if best_insertion_info:
                    farm_to_add, position_to_add = best_insertion_info
                    
                    # Chèn farm vào vị trí tốt nhất đã tìm được
                    cust_list.insert(position_to_add, farm_to_add)
                    
                    # Cập nhật trạng thái
                    assigned_farms.add((day_idx, farm_to_add))
                    residual_capacity -= _resolve_farm_for_ci_local(farm_to_add)[1]
                    new_finish_time = best_overall_finish
                    candidate_farms.discard(farm_to_add)
                    improved = True # Đặt cờ để tiếp tục vòng while, thử chèn thêm farm khác
            # <<< KẾT THÚC THAY ĐỔI LỚN >>>


            # Sau khi không thể thêm farm nữa → lưu route vào final_schedule
            # kiểm tra truck có được phép vào tất cả farm trong tuyến + depot
            depot_access = facilities[depot].get('accessibility', None)
            if depot_access is not None and (len(depot_access) <= truck_obj['type_idx'] or depot_access[truck_obj['type_idx']] == 0):
                print(f"🚫 Xe {truck_obj['id']} ({truck_obj['type']}) không được phép vào Depot {depot}")
                continue

            inaccessible_farms = []
            for fid in cust_list:
                _, _, f_info, _ = _resolve_farm_for_ci_local(fid)
                f_acc = f_info.get('accessibility', None)
                if f_acc is not None and (len(f_acc) <= truck_obj['type_idx'] or f_acc[truck_obj['type_idx']] == 0):
                    inaccessible_farms.append(fid)
            if inaccessible_farms:
                print(f"🚫 Xe {truck_obj['id']} không được phép vào các farm: {inaccessible_farms}")
                continue

            final_schedule[day_idx].append((depot, truck, cust_list, chosen_shift))
            # Cập nhật thời gian kết thúc cuối cùng cho truck này trong ngày
            truck_finish_times[day_idx][truck] = (new_finish_time, depot)

            ## LOGIC MỚI: CẬP NHẬT TẢI TRỌNG KHO VÀ KIỂM TRA ĐỂ TẠO CHUYẾN INTER-FACTORY ##
            # 1. Tính tổng lượng sữa của chuyến vừa tạo
            route_total_demand = sum(_resolve_farm_for_ci_local(fid)[1] for fid in cust_list)
            
            # 2. Cập nhật vào bộ đếm tải trọng của kho
            depot_daily_load[day_idx][depot] += route_total_demand
            
            # 3. Kiểm tra xem kho có bị quá tải không
            if depot_daily_load[day_idx][depot] > DEPOT_CAPACITY:
                print(f"    -> 🏭 CẢNH BÁO QUÁ TẢI: Depot {depot} ngày {day_idx} đạt {depot_daily_load[day_idx][depot]:.0f}/{DEPOT_CAPACITY}. Kích hoạt vận chuyển liên kho.")
                
                # Tìm một kho khác trong cùng vùng để chuyển sữa đến
                current_region = facilities[depot]['region']
                candidate_target_depots = [d_idx for d_idx in depots_by_region[current_region] if d_idx != depot]
                
                if candidate_target_depots:
                    # Chọn kho có tải trọng thấp nhất làm kho đích
                    target_depot = min(candidate_target_depots, key=lambda d: depot_daily_load[day_idx][d])
                    
                    # Lượng sữa cần chuyển đi là phần vượt quá sức chứa
                    transfer_amount = depot_daily_load[day_idx][depot] - DEPOT_CAPACITY
                    
                    # Tìm một xe tải phù hợp trong vùng để thực hiện việc vận chuyển
                    transfer_truck = None
                    for t in eligible_trucks_in_region:
                        src_acc = facilities[depot].get('accessibility', None)
                        dst_acc = facilities[target_depot].get('accessibility', None)
                        if (
                            t['capacity'] >= transfer_amount
                            and (src_acc is None or (len(src_acc) > t['type_idx'] and src_acc[t['type_idx']] == 1))
                            and (dst_acc is None or (len(dst_acc) > t['type_idx'] and dst_acc[t['type_idx']] == 1))
                        ):
                            transfer_truck = t
                            break

                    
                    if transfer_truck:
                        # Tạo một "tuyến đường" đặc biệt cho việc vận chuyển liên kho
                        transfer_route_customer = [f'TRANSFER_FROM_{depot}_TO_{target_depot}']
                        # Thêm tuyến này vào lịch trình
                        final_schedule[day_idx].append((depot, transfer_truck['id'], transfer_route_customer, 'INTER-FACTORY'))
                        print(f"        -> 🚚 Tạo chuyến INTER-FACTORY: Xe {transfer_truck['id']} chuyển {transfer_amount:.0f}L từ Depot {depot} đến Depot {target_depot}.")

                        # Cập nhật lại tải trọng của hai kho
                        depot_daily_load[day_idx][depot] -= transfer_amount
                        depot_daily_load[day_idx][target_depot] += transfer_amount
                    else:
                        print(f"        -> ⚠️ KHÔNG TÌM THẤY XE để thực hiện chuyến INTER-FACTORY từ Depot {depot}.")
                else:
                    print(f"        -> ⚠️ KHÔNG CÓ KHO ĐÍCH trong vùng {current_region} để chuyển sữa từ Depot {depot}.")
            ## KẾT THÚC LOGIC MỚI ##

        else:
            # Nếu không tìm được phương án khả thi với bất kỳ xe nào (thời gian) → báo lỗi
            print(f"!!! LỖI THỜI GIAN: Farm {farm_id_to_insert} không thể lên lịch ngày {day_idx}.")

    # KẾT THÚC WHILE → in lịch trình đã tạo
    for day, routes in final_schedule.items():
        print(f"\n📅 Ngày {day}:")
        if not routes:
            print("  (Không có tuyến nào)")
            continue
        # Gom nhóm các tuyến theo truck để in gọn
        truck_routes = defaultdict(list)
        for depot, truck, cust_list, shift in routes:
            truck_routes[truck].append((depot, cust_list, shift))
        for truck, trips in truck_routes.items():
            print(f"  🚚 Truck {truck} chạy {len(trips)} chuyến:")
            for trip_no, (depot, cust_list, shift) in enumerate(trips, 1):
                route_str = " → ".join(str(c) for c in cust_list)  # nối chuỗi ID farm
                ## LOGIC MỚI: Sửa đổi cách in để hiển thị đẹp hơn cho tuyến INTER-FACTORY ##
                if shift == 'INTER-FACTORY':
                    print(f"    🏭 Chuyến đặc biệt ({shift}): {route_str.replace('_', ' ')}")
                else:
                    print(f"    🧭 Chuyến {trip_no} ({shift}) - Depot {depot}: Depot {depot} → {route_str} → Depot {depot}")

    print("\n--- KẾT THÚC COMPUTE_INITIAL_SOLUTION ---")
    print(f"Số nông trại không thể lên lịch: {count}")
    return final_schedule