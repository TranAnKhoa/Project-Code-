import numpy as np
from collections import defaultdict
import random
import re
import copy
from .utils import _clean_base_id
# ======================= HÀM TIỆN ÍCH =======================




def _calculate_route_schedule_and_feasibility_ini(depot_idx, customer_list, shift, start_time_at_depot, problem_instance, truck_info):
    """Kiểm tra tính khả thi của route với time window, đã bao gồm velocity."""
    #Trong hàm này:[0] --> closest depot to customer list, [1] --> customer list i, --> shift: AM/PM, ....
   
    # Nếu danh sách khách rỗng -> kết thúc ngay, trả start_time tại depot (không di chuyển)
    if not customer_list:
        return start_time_at_depot, True
   
    # Lấy các cấu trúc dữ liệu cần thiết từ problem_instance
    dist_matrix = problem_instance['distance_matrix_farms']      # ma trận khoảng cách giữa farms (n x n)
    depot_farm_dist = problem_instance['distance_depots_farms']  # ma trận khoảng cách depot -> farm (m x n)
    farms = problem_instance['farms']                            # danh sách dict mô tả từng farm
    farm_id_to_idx = problem_instance['farm_id_to_idx_map']      # map id -> index tương ứng trong 'farms'
    depot_end_time = 1440  # phút trong ngày (24h * 60) — depot phải về trước thời điểm này
    current_time = start_time_at_depot  # thời gian hiện tại (bắt đầu từ thời điểm xe rời depot)
    truck_name = truck_info['type']      # kiểu xe (ví dụ "Single", "20m", ...)
    # Thiết lập vận tốc tương đối theo kiểu xe — giả lập: "Single" & "Truck and Dog" nhanh (1.0), else 0.5
    # (Ở đây velocity dùng để chia khoảng cách -> thời gian di chuyển.)
    velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5


    # virtual_map chứa các farm bị split (virtual/phantom visits), key là id ảo -> thông tin phần (portion)
    virtual_map = problem_instance.get('virtual_split_farms', {}) #get ra key virtual_split_farms của dict farm


    def _resolve_farm(fid):
        """Hàm con để xử lý farm thật/ảo trong danh sách.


        Trả về:
            base_idx: index của farm thật trong farms (int)
            demand_or_portion: nếu là ảo thì 'portion' (số lượng) else demand thật của farm
            service_time_params: (fix_time, var_param) dùng để tính service_duration
            time_windows: cặp (AM_window, PM_window) — mỗi window là (start, end)
        """
        # Lấy dạng id gốc (loại bỏ hậu tố nếu cần)
        base_id_str = _clean_base_id(fid)
        try:
            # Thử tra bằng key dạng string
            base_idx = farm_id_to_idx[base_id_str]
        except KeyError:
            # Nếu không có, thử convert sang int (nhiều file dùng int keys)
            base_idx = farm_id_to_idx[int(base_id_str)]


        base_info = farms[base_idx]
        # Nếu fid là string và có trong virtual_map -> đây là một farm ảo (tách phần)
        if isinstance(fid, str) and fid in virtual_map:
            portion = virtual_map[fid].get('portion', 0)  # lượng (portion) cần lấy cho visit ảo này
            # Trả về portion thay cho demand
            return base_idx, portion, base_info['service_time_params'], base_info['time_windows']
        else:
            # farm thật -> trả demand đầy đủ
            return base_idx, base_info['demand'], base_info['service_time_params'], base_info['time_windows']


    # ============ xử lý khách đầu tiên (từ depot -> customer đầu) ============
    first_cust_id = customer_list[0]
    # _resolve_farm trả (index_in_farms, demand_or_portion, service_time_params, time_windows)
    first_idx, first_demand, first_params, first_tw = _resolve_farm(first_cust_id)
    # Thời gian di chuyển từ depot tới farm đầu tiên = khoảng cách / vận tốc (velocity)
    travel_time = depot_farm_dist[depot_idx, first_idx] / velocity
    arrival_time = current_time + travel_time  # thời gian đến nơi (chưa tính wait nếu đến sớm)


    # Lấy time window cho shift (shift là 'AM' hoặc 'PM') -> mỗi farm lưu time_windows theo key 'AM'/'PM'
    start_tw, end_tw = first_tw[shift]
    # Nếu đến sau end_tw (quá trễ) -> infeasible
    if arrival_time > end_tw:	
        return -1, False


    # service_start = max(arrival_time, start_tw) -> nếu đến sớm thì chờ đến start_tw
    service_start = max(arrival_time, start_tw)
    fix_time, var_param = first_params
    # service_duration = fix_time + demand / var_param (nếu var_param > 0)
    # lưu ý: var_param thường là tốc độ phục vụ (units per minute). Nếu var_param == 0 -> treat as fix only
    service_duration = fix_time + (first_demand / var_param if var_param > 0 else 0)
    current_time = service_start + service_duration  # cập nhật thời điểm kết thúc phục vụ


    # ============ xử lý các khách tiếp theo (customer_list[1:] ) ============
    for i in range(len(customer_list) - 1):
        # from_idx là index của farm hiện tại (i), to_idx là farm kế tiếp (i+1)
        from_idx, _, _, _ = _resolve_farm(customer_list[i])
        to_idx, to_demand, to_params, to_tw = _resolve_farm(customer_list[i + 1])
        # travel_time giữa 2 farm = ma trận dist_matrix[from, to] / velocity
        travel_time = dist_matrix[from_idx, to_idx] / velocity
        arrival_time = current_time + travel_time


        # lấy time window cho farm kế tiếp ở shift tương ứng
        start_tw, end_tw = to_tw[shift] #shift đc input ở đầu, to_tw = {'AM': [10, 12],'PM': [20, 22]}
        # nếu đến sau end_tw -> không khả thi
        if arrival_time > end_tw:
            return -1, False


        service_start = max(arrival_time, start_tw)
        fix_time, var_param = to_params
        service_duration = fix_time + (to_demand / var_param if var_param > 0 else 0)
        current_time = service_start + service_duration


    # ============ sau khi phục vụ khách cuối, quay lại depot ============
    last_idx, _, _, _ = _resolve_farm(customer_list[-1])
    travel_time_back = depot_farm_dist[depot_idx, last_idx] / velocity
    finish_time_at_depot = current_time + travel_time_back
    # Nếu về depot sau thời gian depot_end_time (1440 phút) -> infeasible
    if finish_time_at_depot > depot_end_time:
        return -1, False


    # Trả về thời gian finish và cờ feasible True
    return finish_time_at_depot, True






# ==================== HÀM CHÍNH (SINGLE-DAY, NÂNG CẤP) ====================
def compute_initial_solution(problem_instance, random_state):
    print("\n--- BÊN TRONG COMPUTE_INITIAL_SOLUTION (SINGLE-DAY, NÂNG CẤP) ---")
    count = 0  # biến đếm số farm không xử lý được (error / infeasible)
    #set: kiểu dữ liệu có thể gồm nhiều type of data --> k có thứ tự, k bị trùng lặp --> như kiểu cái kho để chứa t.tin
    onfly_split_done = set()  # lưu những farm đã bị "on-the-fly split" để không split lại


    # Lấy các cấu trúc chính từ problem_instance --> tất cả biến đầu là dictionary
    farms = problem_instance['farms']                         # list of farm dicts
    facilities = problem_instance['facilities']               # list of depot/facility dicts
    available_trucks = problem_instance['fleet']['available_trucks']  # list của truck dicts
    farm_id_to_idx_map = problem_instance['farm_id_to_idx_map']       # map id -> index


    final_schedule = []  # danh sách kết quả tuyến (mỗi phần tử: (depot, truck, cust_list, shift, start_time))  # giới hạn tải trên 1 depot (đơn vị demand)
    depot_capacity=[]
    farm_demand =[] #dùng để tính median của demnad
    for i in problem_instance['facilities']:
        depot_capacity.append(i['capacity'])
    depot_load = defaultdict(float)  # track tổng demand đã gán cho từng depot (mặc định 0)
    depots_by_region = defaultdict(list)  # map region -> list depot indices
    for i in farms:
        farm_demand.append(i["demand"])
    print(farm_demand)


    # Tạo mapping depots theo region để dùng khi cần chuyển tải giữa depots cùng region
    for i, facility in enumerate(facilities): #i là chỉ số trong dict, facility mới là từng dic theo vòng lặp
        if 'region' in facility:
            depots_by_region[facility['region']].append(i)
    #--> depots_by-region ~ {North: [0,1,2,3], South: [4,5,6,7],...}
    # Danh sách tất cả các farm cần được ghé thăm (lấy id từ farms)
    all_required_visits = [farm['id'] for farm in farms]
    # Xáo trộn thứ tự để lời giải ban đầu có yếu tố ngẫu nhiên
    random_state.shuffle(all_required_visits)


    # truck_finish_times lưu trạng thái finish time của mỗi truck: dict truck_id -> (finish_time, depot_index)
    # khởi tạo default finish_time=0, depot=-1
    truck_finish_times = defaultdict(lambda: (0, -1))


    assigned_farms = set()  # tập các farm đã được gán (đã lên lịch)
    # virtual_map: nơi lưu các farm ảo tạo ra khi split on-the-fly
    virtual_map = problem_instance.setdefault('virtual_split_farms', {})


    def _resolve_farm_for_ci_local(fid):
        """Hàm nội bộ để xử lý farm ảo và tra cứu an toàn.


        Trả về:
            base (str|fid): nếu là ảo thì base là base_id, còn nếu không thì trả nguyên fid
            portion_or_demand: phần demand (nếu ảo) hoặc demand thật
            base_info: dict thông tin farm từ farms[idx]
            idx: index trong farms
        """
        # Nếu fid là str và nằm trong virtual_map -> đây là farm ảo\
        if isinstance(fid, str) and fid in virtual_map:
            base = virtual_map[fid]['base_id']            # id gốc (có thể là string hoặc number)
            portion = virtual_map[fid].get('portion', 0)  # phần demand của visit ảo này
            # Nếu base tiếp tục là ảo (một chuỗi split), lồng while để tìm base thật (base cuối cùng không nằm trong virtual_map)
            while base in virtual_map: #Này chắc không cần dùng tới vì không có scheduling nữa
                base = virtual_map[base]['base_id']
            base_clean = _clean_base_id(base)  # loại bỏ hậu tố nếu base có suffix
            # Tìm index trong map — thử bằng string, nếu không tìm thì cast int
            idx = farm_id_to_idx_map.get(base_clean, farm_id_to_idx_map.get(int(base_clean)))
            base_info = farms[idx] #Toàn bộ dữ liệu của farm[id]
            # Trả về base (gốc), portion, info, index
            return base, portion, base_info, idx


        # Nếu không phải là ảo -> xử lý bình thường
        base_clean = _clean_base_id(fid)
        idx = farm_id_to_idx_map.get(base_clean, farm_id_to_idx_map.get(int(base_clean)))
        base_info = farms[idx]
        return fid, base_info['demand'], base_info, idx


    # ====================== MAIN LOOP ======================
    # Duyệt lần lượt các farm đã xáo trộn
    for i in all_required_visits: #Duyệt i qua n lần của customer
        # Nếu farm đã được gán trước đó (hoặc là 1 phần ảo đã được gán) -> bỏ qua
        if i in assigned_farms:
            continue #Ngay lập tức ngắt i ở hiện tại và move tới i tiếp theo
       
        # Resolve farm (xử lý virtual hoặc base)
        effective_id, eff_demand, farm_details, farm_idx = _resolve_farm_for_ci_local(i) #! Mới đầu i chỉ là index bth thôi mà ?
        # Tìm depot gần nhất: np.argmin trên cột tương ứng farm_idx -> index depot nhỏ nhất về khoảng cách
        closest_depot_idx = int(np.argmin(problem_instance['distance_depots_farms'][:, farm_idx])) # depot nào sẽ được gán cho khách hàng i trong loop (min distance) --> lấy idx của depo
        depot_region = facilities[closest_depot_idx].get('region', None) #Lấy ra regional của depot mới tìm được


        # Map type_to_idx dùng để tương thích với accessibility mask (mảng 4 số)
        type_to_idx = {'Single': 0, '20m': 1, '26m': 2, 'Truck and Dog': 3}
        eligible_trucks_in_region = []
        # Lọc các xe có sẵn trong region đó và có quyền truy cập depot & farm theo type
        for t in available_trucks:
            if t.get('region') != depot_region:
                continue #thoát khỏi xe t và đi tới xe t+1 tiếp theo
           
            # Gắn thêm field t['type_idx'] (chỉ dùng nội bộ) cho tiện truy vấn accessibility
            t['type_idx'] = type_to_idx.get(t.get('type'), -1)
            #Step1: t.get('type') --> "26m", lấy ra type của dictionary thứ t trong danh sách avai trucks
            #Step 2: type_to_idx.get('26m', -1) --> Tra trong dict type_to_idx, nếu "26m" tồn tại trả về idx: 2, nếu không thì idx -1
            #Step 3: --> dict t sẽ có thêm 1 key mới: 'type_idx': 2
            # accessibility tại depot & farm là 1/0 per vehicle type (mảng length 4)
            depot_ok = facilities[closest_depot_idx].get('accessibility', [1]*4)[t['type_idx']] == 1
            #Step 1: facilities[idx] --> lấy key accessibility (nếu không có thì sẽ open full)
            #Step 2: Truy cập vào key type_idx đã tạo ở trên --> [1,1,0,0][2] --> 0 != 1 --> không lấy xe đó --> qua dict của xe t tiếp
           
            farm_ok = farm_details.get('accessibility', [1]*4)[t['type_idx']] == 1
            if depot_ok and farm_ok:
                eligible_trucks_in_region.append(t) #Nếu satisfy acccessibility thì add vào danh sách xe
       
        # Nếu không có xe phù hợp trong region -> in cảnh báo và tiếp tục (count lỗi++)
        if not eligible_trucks_in_region:
            print(f"!!! KHÔNG CÓ XE Ở VÙNG {depot_region} PHÙ HỢP để phục vụ Farm {i}")
            count += 1
            continue
       
        # Tìm công suất lớn nhất trong region (để xem có cần split on-the-fly)
        max_capacity_in_region = max(t['capacity'] for t in eligible_trucks_in_region)
        median_demand = np.median(farm_demand)
        # Nếu demand > max_capacity và farm chưa bị onfly split -> ta sẽ chia farm thành nhiều visits ảo (on-the-fly)
        if eff_demand > max_capacity_in_region and i not in onfly_split_done:
            # Số phần cần chia = ceil(demand / max_capacity)
            num_parts = int(np.ceil(eff_demand / median_demand))
            remaining, true_base = eff_demand, _clean_base_id(effective_id) # (demand,id)
            print(f"⚠️ ON-THE-FLY SPLIT: {i} demand {eff_demand} > {median_demand}. Tạo {num_parts} phần.")
            # Tạo các phần ảo: split_id = f"{i}_onfly_part{k+1}"
            for k in range(num_parts): #Duyệt qua từng demand sau khi bị split
                part_qty = min(median_demand, remaining) #--> lấy median_demand
                split_id = f"{i}_onfly_part{k+1}" #e.g: 7678_onfly_part1
                # Lưu vào virtual_map: base_id là true_base (id gốc trong farms), portion là part_qty
                virtual_map[split_id] = {'base_id': true_base, 'portion': part_qty} #thêm key là "7678_onfly_part1": ....
                # Thêm phần ảo vào danh sách all_required_visits để vòng lặp chính sẽ xét tới
                all_required_visits.append(split_id) #! Thêm các onfly_part này vào cuối array để lặp lại tiếp tục ở trên
                remaining -= part_qty
            # Đánh dấu farm gốc là đã assigned (vì ta đã thay bằng các phần ảo)
            assigned_farms.add(i) # Thêm i vào set farms đã assign (tránh lặp lại)
            onfly_split_done.add(i) #Thêm i vào set farms đã bị split (tránh lặp lại)
            # tiếp tục loop (không cố gắng gán farm gốc nữa)
            continue #Sau khi bị split rồi thì skip farm, quay về vòng for i --> vì chắc chắc demand sẽ thỏa xe
            
        # Nếu đến đây: có ít nhất một xe đủ tải (capacity >= eff_demand) --> #! dành cho farm không bị skip
        suitable_trucks = [t for t in eligible_trucks_in_region if t['capacity'] >= eff_demand] #Hàm này chỉ dùng để check cho farm không bị
        if not suitable_trucks:
            # Lỗi tải trọng: không có xe nào đủ tải (trường hợp này xảy ra nếu không split nhưng demand vẫn lớn)
            print(f"!!! LỖI TẢI TRỌNG: Không có xe đủ tải cho Farm {i} ở vùng {depot_region}.")
            count += 1
            continue
           
        # Tìm phương án tốt nhất (lowest finish_time) giữa các truck và shift
        best_option = (float('inf'), None)  # (finish_time, option_tuple)
        for truck_obj in suitable_trucks:
            truck_id = truck_obj['id']
            last_finish_time, _ = truck_finish_times[truck_id] #Hiện tại thì chưa có xe nào chạy --> (0,-1)
           
            # Nếu truck đã có chuyến trước đó -> start time cho chuyến tiếp theo chậm hơn 30 phút (buffer)
            first_clean_id = _clean_base_id(i)
            farm_id_map = problem_instance['farm_id_to_idx_map']
            first_idx = farm_id_map.get(first_clean_id, farm_id_map.get(int(first_clean_id)))
            first_farm_info = farms[first_idx]
            # Lấy khoảng thời gian phục vụ AM/PM
            velocity = 1.0 if truck_obj in ["Single", "Truck and Dog"] else 0.5
            
            
            # Tính travel_time depot → farm đầu tiên
            travel_time = problem_instance['distance_depots_farms'][closest_depot_idx, first_idx] / velocity
            
            for shift in ['AM', 'PM']: #Duyệt shift = 'AM' và 'PM' và tính thử finish time của khách i
                start_tw = first_farm_info['time_windows'][shift][0]
                start_time = last_finish_time + 30 if last_finish_time > 0 else max(0, start_tw - travel_time) #Nếu chuyến đi đầu trong ngày --> khởi hành từ lúc vừa kịp để k đợi wait 
                # Gọi hàm kiểm tra lịch & feasibility cho route chỉ chứa 1 customer (i)
                finish_time, feasible = _calculate_route_schedule_and_feasibility_ini(
                    closest_depot_idx, [i], shift, start_time, problem_instance, truck_obj
                )
                # Nếu feasible và finish_time nhỏ hơn best_option -> update best_option
                if feasible and finish_time < best_option[0]:
                    best_option = (finish_time, (closest_depot_idx, truck_id, [i], shift, start_time, truck_obj))


        # Nếu không tìm được phương án (best_option[1] None) => lỗi thời gian (không reasanable shift/time)
        if best_option[1] is None:
            print(f"!!! LỖI THỜI GIAN: Farm {i} không thể lên lịch.")
            continue


        # Nếu tìm được option tốt -> unpack
        _, (depot, truck, cust_list, chosen_shift, chosen_start_time, truck_obj) = best_option #Lấy lại thông tin từ best_option để record
        # Gán farm(s) cho assigned set
        assigned_farms.update(cust_list) #cập nhật assiged farm
        new_finish_time = best_option[0]
        # Cập nhật finish time cho truck
        truck_finish_times[truck] = (new_finish_time, depot)
        # Tính tổng demand trên route này (nếu cust_list có nhiều phần ảo) — dùng _resolve_farm_for_ci_local để tính
        route_total_demand = sum(_resolve_farm_for_ci_local(fid)[1] for fid in cust_list)
        depot_load[depot] += route_total_demand #cập nhật depot_load


        # ✅ Thêm start_time vào final_schedule (bao gồm shift)
        final_schedule.append((depot, truck, cust_list, chosen_shift, chosen_start_time))


        # --- Xử lý quá tải depot ---
        # Nếu depot_load vượt DEPOT_CAPACITY -> cố gắng transfer tới depot khác trong cùng region
        if depot_load[depot] > depot_capacity[depot]:
            print(f"    -> 🏭 CẢNH BÁO QUÁ TẢI: Depot {depot} đạt {depot_load[depot]:.0f}/{depot_capacity[depot]}.")
            current_region = facilities[depot]['region']
            # candidate_target_depots: các depot khác cùng region (trừ depot hiện tại)
            candidate_target_depots = [d_idx for d_idx in depots_by_region[current_region] if d_idx != depot] # Tao arr candidate cua cac depot trong region khac
            transfer_truck = None


            if candidate_target_depots:
                # Chọn target_depot có depot_load nhỏ nhất
                target_depot = min(candidate_target_depots, key=lambda d: depot_load[d])
                transfer_amount = depot_load[depot] - depot_capacity[depot]  # amount cần chuyển


            # Tìm một truck sẵn có trong region có thể chở transfer_amount và có accessibility ở cả hai depot
            # --- [BẮT ĐẦU KHỐI LOGIC MỚI] ---
            # Logic mới: Biến transfer thành một "chuyến đi thật"
            # 1. Lấy ma trận khoảng cách giữa các depot (Giả định)
            depot_dist_matrix = problem_instance.get('distance_matrix_facilities')
            # 2. Lấy khoảng cách di chuyển thực tế
            travel_dist = depot_dist_matrix[depot][target_depot]
            # 3. Tìm xe phù hợp nhất (cả xe rảnh và xe multi-trip)
            best_truck_option = (None, float('inf'), 0) # (truck_obj, finish_time, start_time)
            for t in available_trucks:
                # ----- BỘ LỌC CƠ BẢN -----
                # A. Lọc region
                if t.get('region') != depot_region:
                    continue
                    
                # B. Lọc tải trọng
                if t['capacity'] < transfer_amount:
                    continue

                # C. Lọc accessibility (khả năng truy cập)
                t_type_idx = type_to_idx.get(t.get('type'), -1) #nếu t.get('type') không có sẽ trả về -1 --> bị rảnh 
                if t_type_idx == -1: continue # Bỏ qua nếu type xe không xác định

                src_acc = facilities[depot].get('accessibility', [1]*4)
                dst_acc = facilities[target_depot].get('accessibility', [1]*4)
                
                if not (src_acc[t_type_idx] == 1 and dst_acc[t_type_idx] == 1):
                    continue
                
                # ----- TÍNH TOÁN THỜI GIAN THỰC -----
                # D. Tính vận tốc và thời gian di chuyển
                truck_name = t['type']
                velocity = 1.0 if truck_name in ["Single", "Truck and Dog"] else 0.5

                travel_time = travel_dist / velocity
                
                # E. Tìm thời điểm xe có thể bắt đầu (earliest start)
                # Lấy finish_time cuối cùng của xe, nếu không có là 0
                last_finish_time, _ = truck_finish_times[t['id']] 
                
                # Thêm 30 phút buffer (giống như khi đi farm)
                earliest_start_time = last_finish_time + 30 if last_finish_time > 0 else 0
                
                # F. Tính thời điểm kết thúc
                new_finish_time = earliest_start_time + travel_time
                
                # G. Kiểm tra tính khả thi (phải về trước 1440 phút)
                if new_finish_time < 1440:
                    # Nếu khả thi, kiểm tra xem có phải là lựa chọn tốt nhất (kết thúc sớm nhất)
                    if new_finish_time < best_truck_option[1]:
                        best_truck_option = (t, new_finish_time, earliest_start_time)

            # 4. Sau khi duyệt hết các xe, gán tuyến nếu tìm được
            transfer_truck, final_finish_time, trip_start_time = best_truck_option

            if transfer_truck:
                # Đã tìm thấy xe tốt nhất
                transfer_route_customer = [f'TRANSFER_FROM_{depot}_TO_{target_depot}']
                
                # Thêm vào lịch trình với thời gian bắt đầu đã tính
                final_schedule.append(
                    (depot, transfer_truck['id'], transfer_route_customer, 'INTER-FACTORY', trip_start_time)
                )
                
                # Cập nhật finish_time "thật" cho xe tải
                truck_finish_times[transfer_truck['id']] = (final_finish_time, target_depot)
                
                # Cập nhật tải trọng
                depot_load[depot] -= transfer_amount
                depot_load[target_depot] += transfer_amount
                
                print(f"    -> 🚚 Tạo chuyến INTER-FACTORY ({depot}->{target_depot}) thành công. "
                    f"Xe {transfer_truck['id']} (Bắt đầu: {trip_start_time:.0f}, Kết thúc: {final_finish_time:.0f})")
            
            else:
                # Không tìm được xe nào (vì không đủ tải, không rảnh, hoặc không đủ thời gian trong ngày)
                print(f"    ⚠️ Không có xe phù hợp (hoặc đủ thời gian) cho INTER-FACTORY transfer giữa {depot} và {target_depot}.")

            # --- [KẾT THÚC KHỐI LOGIC MỚI] ---


    # ====================== In ra lịch trình kết quả (tổng quan) ======================
    print("\n📅 LỊCH TRÌNH CHO NGÀY:")
    if not final_schedule:
        print("  (Không có tuyến nào)")
    else:
        # Gom trips theo truck để in gọn
        truck_routes = defaultdict(list) #cái này như kiểu array []- hay vãi ~
        for depot, truck, cust_list, shift, start_time in final_schedule:
            truck_routes[truck].append((depot, cust_list, shift, start_time)) #add tuyến xe theo từng xe vào
        for truck, trips in truck_routes.items():
            print(f"  🚚 Truck {truck} chạy {len(trips)} chuyến:")
            for trip_no, (depot, cust_list, shift, start_time) in enumerate(trips, 1): #1 là số bước bắt đầu --> sẽ đếm là trip 1,2... thay vì 0
                route_str = " → ".join(str(c) for c in cust_list)
                if shift == 'INTER-FACTORY':
                    # Chuyến đặc biệt inter-factory: cust_list là tên giả 'TRANSFER_FROM_a_TO_b'
                    print(f"    🏭 Chuyến đặc biệt ({shift}): {route_str.replace('_', ' ')}")
                else:
                    # Chuyển start_time (phút) -> hour:minute cho in ra thân thiện
                    h, m = divmod(int(start_time), 60)
                    print(f"    🧭 Chuyến {trip_no} ({shift}) - Depot {depot} (Xuất phát {h:02d}:{m:02d}): Depot {depot} → {route_str} → Depot {depot}")


    print("\n--- KẾT THÚC COMPUTE_INITIAL_SOLUTION ---")
    print(f"Số nông trại không thể lên lịch: {count}")
    return final_schedule