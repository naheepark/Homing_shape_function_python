import glob
import matplotlib.pyplot as plt
import os
import numpy as np

# 저장 폴더 설정
save_folder = "vector_"
os.makedirs(save_folder, exist_ok=True)

# contour_*.gen 파일을 모두 찾음
file_list = glob.glob("contour_*.gen")

for file_path in file_list:
    x, y = [], []
    
    # 파일에서 좌표 읽기
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[1:-2]  # 첫 줄 제거, 마지막 두 줄 제거
    
    for i, line in enumerate(lines, start=2):
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
            except ValueError:
                print(f"숫자 변환 실패 (줄 {i}) - {line.strip()}")

    # numpy 배열로 변환
    x = np.array(x)
    y = np.array(y)

    # 시계방향 45도 회전
    theta = -np.pi / 4
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    # 회전된 좌표 튜플
    rotated_points = list(zip(x_rot, y_rot))

    # y값이 가장 큰 점 찾기 (12시 방향)
    middle_point = max(rotated_points, key=lambda p: p[1])
    
    
    # x좌표 기준으로 정렬된 점 번호 매기기
    sorted_points = sorted(rotated_points, key=lambda p: p[0])

    angle_changes = []

    for i in range(2, len(sorted_points) - 2):
        # 벡터 A: i-2 → i-1
        p1 = np.array(sorted_points[i-2])
        p2 = np.array(sorted_points[i-1])
        v1 = p2 - p1

        # 벡터 B: i+1 → i+2
        p3 = np.array(sorted_points[i+1])
        p4 = np.array(sorted_points[i+2])
        v2 = p4 - p3

        # 정규화
        norm_v1 = v1 / np.linalg.norm(v1)
        norm_v2 = v2 / np.linalg.norm(v2)

        # 각도 계산
        dot = np.clip(np.dot(norm_v1, norm_v2), -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
        
        # if angle_deg > 2:
        #     angle_changes.append((i, angle_deg))

        angle_changes.append((i, angle_deg))
        
    # vector_max = max(angle_changes, key = lambda x : x[1])
    # print(vector_max)
    
    max_angle_changes_x = max(angle_changes, key = lambda x:x[1])
    print(max_angle_changes_x)
    # for i in angle_changes:
     
    
    first_derivatives = []
    second_derivatives = []

    for i in range(1, len(sorted_points) - 1):
        x_prev, y_prev = sorted_points[i - 1]
        x_curr, y_curr = sorted_points[i]
        x_next, y_next = sorted_points[i + 1]

        dx = x_next - x_prev
        if dx == 0:
            continue  # 0으로 나누는 것 방지

        # 1차 도함수 (중앙차분)
        dy = y_next - y_prev
        first_derivative = dy / dx
        first_derivatives.append((i, first_derivative))

        # 2차 도함수
        dx_single = x_next - x_curr
        if dx_single == 0:
            continue
        second_derivative = (y_next - 2 * y_curr + y_prev) / (dx_single ** 2)
        second_derivatives.append((i, second_derivative))
             
# --------------------------- 시각화 --------------------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(x_rot, y_rot, 'o-', markersize=3, label='rotated points')
    # plt.plot(middle_point[0], middle_point[1], '*', color='red', markersize=10, label='Max Y Point')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"{file_path} (45도 회전)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 비율 동일하게 설정
    plt.tight_layout()
    
    idx_max, max_angle = max_angle_changes_x
    x_star, y_star = sorted_points[idx_max]
    #벡터 활용
    plt.plot(x_star, y_star, marker='*', color='blue', markersize=12, label='Max Angle Point')

    # 곡률(2차 미분) 활용
    max_curvature = max(second_derivatives, key=lambda x: abs(x[1]))
    idx_curv, _ = max_curvature
    x_curv, y_curv = sorted_points[idx_curv]

    plt.plot(x_curv, y_curv, marker='o', color='purple', markersize=10, label='Max Curvature')
    base_name = os.path.splitext(os.path.basename(file_path))[0]   
    
    ## 따로그리기
    # 1차 도함수 시각화 (기울기 변화)
    plt.figure(figsize=(10, 4))
    plt.plot([i for i, _ in first_derivatives],
            [val for _, val in first_derivatives],
            label="1st Derivative (Gradient)", color='orange')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Index")
    plt.ylabel("dy/dx")
    plt.title(f"{file_path} - 1st Derivative")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{base_name}_1st_derivative.png"), dpi=300)
    plt.close()


    # 2차 도함수 시각화 (곡률)
    plt.figure(figsize=(10, 4))
    plt.plot([i for i, _ in second_derivatives],
            [val for _, val in second_derivatives],
            label="2nd Derivative (Curvature)", color='purple')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Index")
    plt.ylabel("d²y/dx²")
    plt.title(f"{file_path} - 2nd Derivative (Curvature)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{base_name}_2nd_derivative.png"), dpi=300)
    plt.close()
    

    

    # 데이터 준비
    x1 = [i for i, _ in first_derivatives]
    y1 = [val for _, val in first_derivatives]

    x2 = [i for i, _ in second_derivatives]
    y2 = [val for _, val in second_derivatives]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # 첫 번째 x축 (아래쪽), y축 공통
    ax1.plot(x1, y1, label="1st Derivative (dy/dx)", color='orange')
    ax1.set_xlabel("Index for 1st Derivative (Bottom X-Axis)")
    ax1.set_ylabel("Value")
    ax1.tick_params(axis='x', labelcolor='orange')
    ax1.tick_params(axis='y', labelcolor='black')

    # 두 번째 x축 (위쪽)
    ax2 = ax1.twiny()
    ax2.plot(x2, y2, label="2nd Derivative (d²y/dx²)", color='purple')
    ax2.set_xlabel("Index for 2nd Derivative (Top X-Axis)")
    ax2.tick_params(axis='x', labelcolor='purple')

    # 제목 및 범례
    fig.suptitle(f"{file_path} - Shared Y, Dual X Axes")
    fig.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{base_name}_dual_xaxes.png"), dpi=300)
    plt.close()

    # x축 (공통)
    x_vals_1st = [i for i, _ in first_derivatives]
    y_vals_1st = [val for _, val in first_derivatives]

    x_vals_2nd = [i for i, _ in second_derivatives]
    y_vals_2nd = [val for _, val in second_derivatives]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # 왼쪽 y축: 1차 도함수
    color1 = 'orange'
    ax1.set_xlabel("Index")
    ax1.set_ylabel("1st Derivative (dy/dx)", color=color1)
    ax1.plot(x_vals_1st, y_vals_1st, color=color1, label='1st Derivative')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(0, color='gray', linestyle='--')

    # 오른쪽 y축: 2차 도함수
    ax2 = ax1.twinx()
    color2 = 'purple'
    ax2.set_ylabel("2nd Derivative (Curvature)", color=color2)
    ax2.plot(x_vals_2nd, y_vals_2nd, color=color2, label='2nd Derivative')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(0, color='gray', linestyle='--')

    # 제목 및 레이아웃
    plt.title(f"{file_path} - 1st and 2nd Derivative (Twin Axes)")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "combined_derivatives.png"), dpi=300)
    plt.close()


    # ## 겹쳐그리기
    # fig, ax1 = plt.subplots(figsize=(10, 4))

    # # --- 첫 번째 y축: 1차 도함수 ---
    # x1 = [i for i, _ in first_derivatives]
    # y1 = [val for _, val in first_derivatives]
    # line1 = ax1.plot(x1, y1, label="1st Derivative (dy/dx)", color='orange')
    # ax1.set_ylabel("dy/dx", color='orange')
    # ax1.tick_params(axis='y', labelcolor='orange')

    # # --- 두 번째 y축: 2차 도함수 (곡률) ---
    # ax2 = ax1.twinx()
    # x2 = [i for i, _ in second_derivatives]
    # y2 = [val for _, val in second_derivatives]
    # line2 = ax2.plot(x2, y2, label="2nd Derivative (d²y/dx²)", color='purple')
    # ax2.set_ylabel("d²y/dx²", color='purple')
    # ax2.tick_params(axis='y', labelcolor='purple')

    # # --- 공통 x축 ---
    # ax1.set_xlabel("Index")
    # ax1.axhline(0, color='gray', linestyle='--', linewidth=1)  # x축 기준선은 왼쪽 축(ax1)에만 넣어도 됨

    # # --- 범례 결합 ---
    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper left')

    # # --- 제목 및 저장 ---
    # plt.title(f"{file_path} - Derivatives Combined (Dual Y-Axis)")
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_folder, f"{base_name}_combined_derivatives.png"), dpi=300)
    # plt.close()


    # 겹쳐그리는법
    # # 공통 X축 인덱스를 기준으로 함께 그리기
    # plt.figure(figsize=(10, 4))

    # # 1차 도함수
    # plt.plot(
    #     [i for i, _ in first_derivatives],
    #     [val for _, val in first_derivatives],
    #     label="1st Derivative (dy/dx)", 
    #     color='orange'
    # )

    # # 2차 도함수
    # plt.plot(
    #     [i for i, _ in second_derivatives],
    #     [val for _, val in second_derivatives],
    #     label="2nd Derivative (d²y/dx²)", 
    #     color='purple'
    # )

    # # 기준선
    # plt.axhline(0, color='gray', linestyle='--')

    # # 시각화 설정
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.title(f"{file_path} - Derivatives Combined")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # # 저장
    # plt.savefig(os.path.join(save_folder, f"{base_name}_combined_derivatives.png"), dpi=300)
    # plt.close()


    # 저장
    save_path = os.path.join(save_folder, f"{base_name}_rotated.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"→ 저장 완료: {save_path}")