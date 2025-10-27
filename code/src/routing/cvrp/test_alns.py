import sys
import os

# --- THÊM ĐƯỜNG DẪN ĐỂ PYTHON TÌM THẤY MODULE 'routing' ---
# Đoạn code này đảm bảo các lệnh import bên dưới hoạt động chính xác
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- IMPORT CÁC THƯ VIỆN VÀ HÀM CẦN THIẾT ---
try:
    from routing.cvrp.alns_cvrp import cvrp_helper_functions
    from routing.cvrp.alns_cvrp.cvrp_env import cvrpEnv 
    from routing.cvrp.alns_cvrp.initial_solution import compute_initial_solution
    from routing.cvrp.alns_cvrp.destroy_operators import random_removal, worst_removal 
    from routing.cvrp.alns_cvrp.repair_operators import best_insertion, regret2_insertion
    from numpy.random import RandomState
    import numpy as np # Import numpy để sử dụng
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Vẫn bị lỗi Import: {e}")
    sys.exit()

# --- SETUP ---
# Sử dụng đường dẫn tương đối để code linh hoạt hơn
# Giả sử cấu trúc là: Project Code/code/src/... và Project Code/output_data/...
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
INSTANCE_FILE = os.path.join(base_path, 'output_data', 'CEL_structured_instance.pkl')
SEED = 1234
ITERATIONS = 100

print(f"Đang đọc instance từ: {INSTANCE_FILE}")

# --- 1. ĐỌC DỮ LIỆU ---
# SỬA ĐỔI QUAN TRỌNG: Giải nén tuple trả về từ read_input_cvrp
(nb_customers, capacity, dist_matrix, dist_depots, demands, 
 cus_st, cus_tw, depot_tw, problem_instance_dict) = cvrp_helper_functions.read_input_cvrp(INSTANCE_FILE)

random_state = RandomState(SEED)

# --- 2. TẠO LỜI GIẢI BAN ĐẦU ---
# SỬA ĐỔI QUAN TRỌNG: Truyền đúng dictionary vào hàm
initial_schedule = compute_initial_solution(problem_instance_dict, random_state)

# --- THÊM DÒNG NÀY ĐỂ KIỂM TRA ---
print("\n--- LỜI GIẢI BAN ĐẦU (INITIAL SCHEDULE) ---")
print(initial_schedule)
print("-----------------------------------------\n")


# SỬA ĐỔI QUAN TRỌNG: Khởi tạo cvrpEnv với các tham số đúng
current_solution = cvrpEnv(
    initial_schedule=initial_schedule, 
    problem_instance=problem_instance_dict, 
    seed=SEED,
    # Truyền các tham số cũ để tương thích ngược
    nb_customers=nb_customers, truck_capacity=capacity, dist_matrix_data=dist_matrix,
    dist_depot_data=dist_depots, demands_data=demands, customer_st=cus_st,
    customer_tw=cus_tw, depot_tw=depot_tw
)

best_solution = current_solution
best_obj, _ = best_solution.objective()
print(f"Initial Objective: {best_obj:.2f}")

# --- 3. VÒNG LẶP ALNS ĐƠN GIẢN ---
for i in range(ITERATIONS):
    # Chọn ngẫu nhiên 1 toán tử destroy và 1 repair
    destroyed_solution, unvisited_customers = random_removal(current_solution, random_state)
    
    # SỬA ĐỔI QUAN TRỌNG: Truyền unvisited_customers vào hàm repair
    repaired_solution = best_insertion(destroyed_solution, random_state, unvisited_customers=unvisited_customers)

    current_obj, _ = current_solution.objective()
    repaired_obj, _ = repaired_solution.objective()

    # Tiêu chuẩn chấp nhận đơn giản (chỉ chấp nhận lời giải tốt hơn)
    if repaired_obj < current_obj:
        current_solution = repaired_solution
        if repaired_obj < best_obj:
            best_solution = repaired_solution
            best_obj = repaired_obj
            print(f"Iteration {i}: New best found! Objective: {best_obj:.2f}")
    # (Bạn có thể thêm logic Simulated Annealing ở đây nếu muốn)

print(f"\nFinal Best Objective: {best_solution.objective()[0]:.2f}")