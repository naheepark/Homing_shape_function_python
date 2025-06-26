import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from matplotlib.patches import Ellipse

save_folder = "debugging_" #! <--------------------------- 폴더명 설정
os.makedirs(save_folder, exist_ok=True)

file_list = glob.glob("contour_*.gen")

# ----------------------------------------------------------------------------------------------

def fit_line_least_squares(points):
    x = np.array([pt[0] for pt in points])
    y = np.array([pt[1] for pt in points])
    m, b = np.polyfit(x, y, deg=1)  # 1차 다항식 (직선)
    return m, b  # 직선 y = m*x + b

#* 최소자승 기반 타원 피팅
def fit_ellipse_least_squares(points):
    if len(points) < 8:
        raise ValueError("최소 6개의 점이 필요합니다.")
        
    X = []
    Y = []

    for x, y in points:
        X.append([x**2, x*y, y**2, x, y, 1])  # A, B, C, D, E, F
        Y.append(0)  # 식: Ax² + Bxy + Cy² + Dx + Ey + F = 0

    X = np.array(X)
    Y = np.array(Y)

    coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)  
    # np.linalg.lstsq: numpy에서 제공하는 선형방정식 최소자승법 계산 함수
    #점의 개수가 변수보다 많아지기 때문에 과잉결정 시스템에서 오차를 최소화함
    #residuals: 잔차제곱합, 작을수록 모델이 잘 맞음
    #행렬 X의 유효랭크
    #s: 행렬 X의 특이갑
    return coeffs  # A, B, C, D, E, F

def ellipse_equation_function(A, B, C, D, E, F):
    def f(x, y):
        return A * x**2 + B * x * y + C * y**2 + D * x + E * y + F
    return f

for file_path in file_list:
    x, y = [], []
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
    # numpy 변환
    x = np.array(x)
    y = np.array(y)
    

# #* align(시계방향 45도 회전)
#     theta = -np.pi / 4  # -45도
#     x_rot = x * np.cos(theta) - y * np.sin(theta)
#     y_rot = x * np.sin(theta) + y * np.cos(theta)
#     # (x, y) 튜플 리스트 생성
#     # 회전된 좌표를 (x, y) 튜플 리스트로 만들고, float으로 변환
#     rotated_points = [(float(px), float(py)) for px, py in zip(x_rot, y_rot)]

#     print(f"{file_path} 좌표 수: {len(rotated_points)}")
#     print(rotated_points[:5])  # 앞 5개 미리 보기
    
#     # rotated_points는 [(x, y), (x, y), ...] 형태의 리스트
#     middle_point = max(rotated_points, key=lambda p: p[1])
#     print("y값이 가장 큰 점:", middle_point)
#     middle_x_point = middle_point[0]
#     print(middle_x_point)
    
#* x축 기준 최소/최대 점 구하기
    min_point = min(rotated_points, key=lambda p: p[0])
    max_point = max(rotated_points, key=lambda p: p[0])

    print("x값이 가장 작은 점:", min_point)
    print("x값이 가장 작은 점의 x값:", min_point[0])
    print("x값이 가장 큰 점:", max_point)
    print("x값이 큰 점의 x값:", max_point[0])

    # #구간 조절 #! <------------------------------------------------------ 구간 변수 조절
    # def section(start_point, end_x):
    #     start_x = start_point[0]
    #     delta = end_x - start_x
    #     a = start_x + delta * (1/3)
    #     b = start_x + delta * (2/3)
    #     return a, b

    
    # def section_1(min_point, middle_x_point):
    #     k = middle_x_point - min_point[0]
    #     a = min_point[0] + (k * (1 / 3))
    #     b = min_point[0] + (k * (2 / 3)) 
    #     return a, b
        
        
    # def section_2(middle_x_point, max_point):
    #     k = max_point[0] - middle_x_point
    #     a = middle_x_point + (k * (1 / 3))
    #     b = middle_x_point + (k * (2 / 3))
    #     return a, b
    
        # a1, a2, a3, a4 에 가장 가까운 점

    
    print(section(min_point, middle_point[0]), section(middle_point, max_point[0]))
    a1, a2 = section(min_point, middle_point[0])
    a3, a4 = section(middle_point, max_point[0])
    
    a1_nearest = min(rotated_points, key=lambda p: abs(p[0] - a1))
    a2_nearest = min(rotated_points, key=lambda p: abs(p[0] - a2))
    a3_nearest = min(rotated_points, key=lambda p: abs(p[0] - a3))
    a4_nearest = min(rotated_points, key=lambda p: abs(p[0] - a4))

    print("a1에 가장 가까운 점:", a1_nearest)
    print("a2에 가장 가까운 점:", a2_nearest)
    print("a3에 가장 가까운 점:", a3_nearest)
    print("a4에 가장 가까운 점:", a4_nearest)
    
    
#* 최소자승법으로 좌측, 우측 선분 표시 (1)
    def fit_line_least_squares_function(points):
        #입력된 다수의 (x, y) 점들을 기반으로 최소제곱 직선 y = mx + b를 구하고, 이를 함수 형태로 반환합니다.
        x = np.array([pt[0] for pt in points])
        y = np.array([pt[1] for pt in points])
        
        # 최소제곱법을 통한 계수 계산
        m, b = np.polyfit(x, y, deg=1)
        
        # 함수 객체 반환
        def f(x_input):
            return m * x_input + b

        return f
        
    x_start, x_end = sorted([a1, a2])
    a1_a2_points = [p for p in rotated_points if x_start <= p[0] <= x_end]
    Left_lf = fit_line_least_squares_function(a1_a2_points) if a1_a2_points else None

    x_start, x_end = sorted([a3, a4])
    a3_a4_points = [p for p in rotated_points if x_start <= p[0] <= x_end]
    Right_lf = fit_line_least_squares_function(a3_a4_points) if a3_a4_points else None

    

#* a1, a2기준 왼쪽 선분, a3, a4 기준 오른쪽 선분 표시 (1)
    def Linear(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            raise ValueError("기울기를 구할 수 없음")
        m = (y2 - y1) / (x2 - x1)  # 기울기
        b = y1 - m * x1            # y절편
        
        def f(x):
            return m * x + b
        
        return f 

    # 두 점을 지나는 선형함수 생성
    Left_Linear_function = Linear(a1_nearest, a2_nearest)
    Right_Linear_function = Linear(a3_nearest, a4_nearest)
    
       

#*곡선 위에 랜덤 점 찍기      
    # 구간 내 점 (a2 ≤ x ≤ a3)
    a2_a3_points = [p for p in rotated_points if a2 <= p[0] <= a3]
    print(f"총 {len(a2_a3_points)}개의 점이 a2({a2:.3f}), a3({a3:.3f}) 사이에 있습니다.")

    sample_list = []
    # 샘플 생성
    for i in range(100): #! <---------------------------------------- 테스트 케이스 및 점 개수 설정
        if len(a2_a3_points) >= 8:
            sample = random.sample(a2_a3_points, 8)
            sample_list.append(sample)

 #* 타원
 
    # def fit_ellipse_through_5_points(points):
    #     #주어진 5개 점을 정확히 통과하는 타원의 계수 A,B,C,D,E,F를 계산합니다. F는 -1로 고정합니다.
    #     if len(points) != 5:
    #         raise ValueError("점은 정확히 5개여야 합니다.")
    #     X = []
    #     Y = []

    #     for x, y in points:
    #         X.append([x**2, x*y, y**2, x, y])  # A, B, C, D, E
    #         Y.append(1)  # F = -1 → 식 양변을 +1로

    #     X = np.array(X)
    #     Y = np.array(Y)

        
    #     coeffs = np.linalg.solve(X, Y)  # 정확히 5개 점이므로 정방행렬 (5x5), 방정식과 변수의 개수를 맞춤
    #     A, B, C, D, E = coeffs
    #     F = -1
    #     return A, B, C, D, E, F
    
    
    # def ellipse_equation_function(A, B, C, D, E, F):
    #     # 타원 일반 방정식을 함수 형태로 반환: f(x, y) = Ax² + Bxy + Cy² + Dx + Ey + F
    #     def f(x, y):
    #         return A * x**2 + B * x * y + C * y**2 + D * x + E * y + F
    #     return f

#* 타원 검증 후 타원들 중 넓이가 가장 작은 타원을 찾기
    best_area = float('inf') # 최솟값을 찾기 위한 전형적인 패턴
    best_func = None
    best_sample = None
    best_coeffs = None  # A, B, C 등 저장용

    for sample in sample_list:
        # A, B, C, D, E, F = fit_ellipse_through_5_points(sample)
        A, B, C, D, E, F = fit_ellipse_least_squares(sample)
        # 타원 조건 검사
        # if B**2 - 4*A*C >= 0:
        #     print("타원이 아닙니다: B² - 4AC ≥ 0")
        #     continue  # 타원이 아니므로 스킵
        
        ellipse = ellipse_equation_function(A, B, C, D, E, F)

        x_vals = [pt[0] for pt in sample]
        y_vals = [pt[1] for pt in sample]
        width = max(x_vals) - min(x_vals)
        height = max(y_vals) - min(y_vals)
        area = np.pi * (width / 2) * (height / 2)

        if area < best_area:
            best_area = area
            best_func = ellipse
            best_sample = sample
            best_coeffs = (A, B, C, D, E, F)
            
            
    #만약 best_sample에 할당된 값이 없다는건 타원이 만들어 지지 않아서 타원 조건 검사에서 모두 스킵이 되어 버림


    
 # ---------------------------------------시각화-------------------------------------------
    

    # 회전된 그래프 그리기
    plt.figure()
    plt.plot(x_rot, y_rot, marker='o', label='rotated points', markersize=3)

    # y값이 가장 큰 점을 빨간 별표로 표시
    plt.plot(middle_point[0], middle_point[1], marker='*', color='red', markersize=7, label='Max Y Point')

    plt.title(f"{file_path} (45도 회전)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    
    for x_val in [a1, a2, a3, a4]:
        plt.axvline(x = x_val, color='green', linestyle='--', linewidth=1)

    # section
    for point in [a1_nearest, a2_nearest, a3_nearest, a4_nearest]:
        plt.plot(point[0], point[1], marker='o', color='green', markersize=7)
    
    # 샘플 점 찍기
    for i, sample in enumerate(sample_list):
        x_vals = [pt[0] for pt in sample]
        y_vals = [pt[1] for pt in sample]
        plt.scatter(
            x_vals, y_vals,
            color='orange',
            s=30,                # 점 크기 작게 조정
            label=f'Sample {i+1}',
            zorder=5             # 다른 그래프 요소들보다 위에 그려지도록 설정
    )
        

    # x 범위 설정 (전체 그래프 범위 기준으로 확장)
    x_vals = np.linspace(min(x_rot) - 0.1, max(x_rot) + 0.1, 300)

    # y 값 계산
    y_left = Left_Linear_function(x_vals)
    y_right = Right_Linear_function(x_vals)

    

    # 그래프에 선 추가
    plt.plot(x_vals, y_left, color='orange', linestyle='-', label='Left Linear')
    plt.plot(x_vals, y_right, color='orange', linestyle='-', label='Right Linear')
    
    # 시각화 범위용 x 값 생성
    x_vals = np.linspace(min(x_rot) - 0.1, max(x_rot) + 0.1, 500)

    # 최소제곱 직선 함수 적용
    y_left_lf = Left_lf(x_vals)
    y_right_lf = Right_lf(x_vals)

    # 직선 시각화
    plt.plot(x_vals, y_left_lf, color='red', linestyle='-', label='Left Least Squares')
    plt.plot(x_vals, y_right_lf, color='red', linestyle='-', label='Right Least Squares')


    # 타원 그리드 영역
    x_min = min(x_rot) - 0.1
    x_max = max(x_rot) + 0.1
    y_min = min(y_rot) - 0.1
    y_max = max(y_rot) + 0.1
    
    # # # 가장 작은 타원만 시각화
    # # plt.contour(xx, yy, zz, levels=[0], colors=['orange'], linewidths=2, linestyles='--')
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    # # 타원 판별 조건 확인
    # if B**2 - 4*A*C < 0:
    #     zz = best_func(xx, yy)
    #     plt.contour(xx, yy, zz, levels=[0], colors=['orange'], linewidths=2, linestyles='--')
    # else:
    #     zz = best_func(xx, yy)
    #     plt.contour(xx, yy, zz, levels=[0], colors=['orange'], linewidths=2, linestyles='--')
    #     print(" 타원이 아닙니다: 이 식은 타원 조건을 만족하지 않습니다.")
    
    
    # 안전한 타원 + best_sample 시각화
    if best_func is not None and best_coeffs is not None and best_sample is not None:
        A, B, C, D, E, F = best_coeffs

        # 곡선 시각화
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
        zz = best_func(xx, yy)

        # if B**2 - 4*A*C < 0:
        #     plt.contour(xx, yy, zz, levels=[0], colors=['orange'], linewidths=2, linestyles='--')
        # else:
        #     plt.contour(xx, yy, zz, levels=[0], colors=['orange'], linewidths=2, linestyles='--')
        #     print(f"{file_path}: 타원 조건을 만족하지 않음 (B² - 4AC ≥ 0)")

        # 점 시각화
        x_best = [pt[0] for pt in best_sample]
        y_best = [pt[1] for pt in best_sample]
        plt.scatter(x_best, y_best, color='orange', s=40, label='Best Sample')

    else:
        print(f"{file_path}: 유효한 타원 또는 베스트 샘플이 없어 시각화되지 않음.")

#* 그래프 확대
    # 확대 비율 설정 
    zoom_ratio = 0.95

    # 전체 범위 계산
    x_range = max(x_rot) - min(x_rot)
    y_range = max(y_rot) - min(y_rot)

    x_min_zoom = min(x_rot) + x_range * (0.3 - zoom_ratio / 2)
    x_max_zoom = max(x_rot) - x_range * (0.3 - zoom_ratio / 2)
    y_min_zoom = min(y_rot) + y_range * (0.3 - zoom_ratio / 2)
    y_max_zoom = max(y_rot) - y_range * (0.3 - zoom_ratio / 2)

    # 확대 적용
    plt.xlim(x_min_zoom, x_max_zoom)
    plt.ylim(y_min_zoom, y_max_zoom)
    
    
    

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(save_folder, f"{base_name}_rotated.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"회전 그래프 저장 완료: {output_path}")