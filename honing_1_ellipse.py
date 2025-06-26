import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from matplotlib.patches import Ellipse
# ----------------------------------------항수 정의------------------------------------------------------
#구간 조절 #! <------------------------------------------------------ 변수 조절
def section(start_point, end_x):
    start_x = start_point[0]
    delta = end_x - start_x
    a = start_x + delta * (1/10)
    b = start_x + delta * (9/10)
    return a, b

def fit_line_least_squares_function(points): #최소자승법 직선 근사
        #입력된 다수의 (x, y) 점들을 기반으로 최소제곱 직선 y = mx + b를 구하고, 이를 함수 형태로 반환합니다.
        x = np.array([pt[0] for pt in points])
        y = np.array([pt[1] for pt in points])
        # 최소제곱법을 통한 계수 계산
        m, b = np.polyfit(x, y, deg=1) #np.polyfit: 직선의 방정식을 찾는 함수, deg = 1: 1차다항식, 즉 직선
        # 함수 객체 반환
        def f(x_input):
            return m * x_input + b
        return f

def Linear(p1, p2): #두 개의 점을 확실히 지나는 작선을 계산
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            raise ValueError("기울기를 구할 수 없음")
        m = (y2 - y1) / (x2 - x1)  # 기울기
        b = y1 - m * x1            # y절편
        def f(x):
            return m * x + b
        return f 
    
def fit_ellipse_through_points(points): 
    if len(points) < 6:
        raise ValueError("최소 6개의 점이 필요합니다.")

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    _, _, V = np.linalg.svd(D)
    coeffs = V[-1]  # 작은 특이값 방향

    A, B, C, D_, E, F = coeffs

    # 타원 조건 검사
    if B**2 - 4*A*C >= 0:
        return None

    return A, B, C, D_, E, F

    # 타원 조건 검사 (선택 사항)
    # if B**2 - 4*A*C >= 0:
    #     print("주의: 타원이 아닐 수 있음 (B² - 4AC >= 0)")

    # return A, B, C, D_, E, F

        # # 여러개의 점을 넣으면 최소자승법을 적용한 A, B, C, D, E, F 를 리턴해줌
        # # if len(points) < 100: #! <------------------------------------------------------ 변수 조절
        # #     raise ValueError("최소 100개여야 합니다.")
        
        # X = []
        # Y = []
        # for x, y in points:
        #     X.append([x**2, x*y, y**2, x, y, 1])  # A, B, C, D, E, F
        #     Y.append(0)
        # X = np.array(X)
        # Y = np.array(Y)
        # coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None) # 선형방적식에 최소자승법을 적용한 후 계수만 구함
        # # linalg: linear algebra, lstsq: (least squares) 최소자승법
        # # coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None) 로 4개의 값이 나오지만 *_로 필요없는 나머지는 버림
        # A, B, C, D, E, F = coeffs
        # return A, B, C, D, E, F
    
def ellipse_equation_function(A, B, C, D, E, F):
    # 타원 일반 방정식을 함수 형태로 반환: f(x, y) = Ax² + Bxy + Cy² + Dx + Ey + F
    def f(x, y):
        return A * x**2 + B * x * y + C * y**2 + D * x + E * y + F
    return f    

def cov_points_colinear(points, threshold=1e-4):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # 2차원 공분산 행렬의 고유값
    cov = np.cov(x, y)
    eigvals = np.linalg.eigvalsh(cov) # 반환값은 A의 고유값들로 1차원 배열
    # 고유값 중 작은 값이 거의 0이라면 직선에 가까움
    return eigvals[0] < threshold

def remove_noise(points, threshold=5): #이상치 제거
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    x_mean, y_mean = np.mean(x), np.mean(y)
    x_std, y_std = np.std(x), np.std(y) # std(standard deviation): 표준편차

    filtered = [
        p for p in points 
        if abs(p[0] - x_mean) < threshold * x_std and abs(p[1] - y_mean) < threshold * y_std
    ]
    return filtered #이상치가 제거된 점들이 들어감

 # ----------------------------------------------------------------------------------------------   
save_folder = "ellipse_" #! <--------------------------- 폴더명 설정
os.makedirs(save_folder, exist_ok=True)
file_list = glob.glob("contour_*.gen")

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

#* align(시계방향 45도 회전)
    theta = -np.pi / 4  # -45도
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    
    # (x, y) 튜플 리스트 생성
    # 회전된 좌표를 (x, y) 튜플 리스트로 만들고, float으로 변환
    rotated_points = [(float(px), float(py)) for px, py in zip(x_rot, y_rot)]

    print(f"{file_path} 좌표 수: {len(rotated_points)}")
    print(rotated_points[:5])  # 앞 5개 미리 보기
    
    # rotated_points는 [(x, y), (x, y), ...] 형태의 리스트
    middle_point = max(rotated_points, key=lambda p: p[1])
    print("y값이 가장 큰 점:", middle_point)
    middle_x_point = middle_point[0]
    print(middle_x_point)
       
#* x축 기준 최소/최대 점 구하기
    min_point = min(rotated_points, key=lambda p: p[0])
    max_point = max(rotated_points, key=lambda p: p[0])

    print("x값이 가장 작은 점:", min_point)
    print("x값이 가장 작은 점의 x값:", min_point[0])
    print("x값이 가장 큰 점:", max_point)
    print("x값이 큰 점의 x값:", max_point[0])
    
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
    
    
#* 최소자승법으로 좌측, 우측 선분 표시
    a1_a2_points = [p for p in rotated_points if a1 <= p[0] <= a2]
    Left_lf = fit_line_least_squares_function(a1_a2_points)
    
    a3_a4_points = [p for p in rotated_points if a3 <= p[0] <= a4]
    Right_lf = fit_line_least_squares_function(a3_a4_points)
    
#* a1, a2기준 왼쪽 선분, a3, a4 기준 오른쪽 선분 표시
    # 두 점을 지나는 선형함수 생성
    Left_Linear_function = Linear(a1_nearest, a2_nearest)
    Right_Linear_function = Linear(a3_nearest, a4_nearest)   
           
#* 이상치 제거   
    # 구간 내 점 (a2 ≤ x ≤ a3)
    a2_a3_points = [p for p in rotated_points if a2 <= p[0] <= a3]
    filtered_points = remove_noise(a2_a3_points, threshold=2.5)
    print(f"총 {len(a2_a3_points)}개의 점이 a2({a2:.3f}), a3({a3:.3f}) 사이에 있습니다.")
    print(f"노이즈 제거 후 타원 피팅 대상 점 개수: {len(filtered_points)}") 
    
    # 곡률 여부 판단용 서브 구간 선택 (예: 중간 1/3 구간)
    sub_a, sub_b = section((a2, 0), a3)  # a2를 시작점으로 section 계산
    curve_points = [p for p in filtered_points if sub_a < p[0] < sub_b]


#* 타원 피팅 
    best_area = float('inf') # 최솟값을 찾기 위한 전형적인 패턴, 처음에는 비교대상이 없기 때문에 무한대로 설정
    best_func = None
    best_sample = None
    best_coeffs = None  # A, B, C, D, E, F 저장용
    
    # 샘플 생성 
    # if not cov_points_colinear(filtered_points):
    for _ in range(200):
        sample = random.sample(filtered_points, 50)

        result = fit_ellipse_through_points(sample)
        if result is None:
            continue  # 타원 조건 불만족 시 건너뛰기

        A, B, C, D, E, F = result


        x_vals = [pt[0] for pt in sample]
        y_vals = [pt[1] for pt in sample]

        width = max(x_vals) - min(x_vals)
        height = max(y_vals) - min(y_vals)
        area = np.pi * (width / 2) * (height / 2)

        if area < best_area:
            best_area = area
            best_sample = sample
            best_coeffs = (A, B, C, D, E, F)
            best_func = ellipse_equation_function(A, B, C, D, E, F)
    # else:
    #     print("filtered_points가 직선 위에 있어 타원 피팅 생략")
            

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
       
       
    x_vals = [pt[0] for pt in best_sample]
    y_vals = [pt[1] for pt in best_sample]

    plt.scatter(
        x_vals, y_vals,
        color='orange',
        s=30,
        label='Best Sample',
        zorder=5)

    # # 샘플 점 찍기
    # for i, sample in enumerate(best_sample):
    #     # x_vals = [pt[0] for pt in sample]
    #     # y_vals = [pt[1] for pt in sample]
    #     # if np.std(x_vals) < 1e-2 or np.std(y_vals) < 1e-2:
    #     #     continue  # 표준편차 필터링
    #     plt.scatter(
    #         x_vals, y_vals,
    #         color='orange',
    #         s=30,                # 점 크기 작게 조정
    #         label=f'Sample {i+1}',
    #         zorder=5)             # 다른 그래프 요소들보다 위에 그려지도록 설정       

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
    
    #  best_sample 시각화
    if best_func is not None and best_coeffs is not None and best_sample is not None:
        # A, B, C, D, E, F = best_coeffs
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400)) #2차원 그리드 틀
        zz = best_func(xx, yy)
        # meshgrid: 2차원 좌표 평면

        min_z = np.min(zz)
        max_z = np.max(zz)

        plt.contour(xx, yy, zz, levels=[0], colors=['orange'], linewidths=2, linestyles='--')
        #타원의 경계선을 그려줌. levels=[0]: f(x, y) = 0인 점들만 연결해서 그림

        x_best = [pt[0] for pt in best_sample]
        y_best = [pt[1] for pt in best_sample]
        plt.scatter(x_best, y_best, color='orange', s=40, label='Best Sample')

# #* 그래프 확대
#     # 확대 비율 설정 
#     zoom_ratio = 0.95

#     # 전체 범위 계산
#     x_range = max(x_rot) - min(x_rot)
#     y_range = max(y_rot) - min(y_rot)

#     x_min_zoom = min(x_rot) + x_range * (0.3 - zoom_ratio / 2)
#     x_max_zoom = max(x_rot) - x_range * (0.3 - zoom_ratio / 2)
#     y_min_zoom = min(y_rot) + y_range * (0.3 - zoom_ratio / 2)
#     y_max_zoom = max(y_rot) - y_range * (0.3 - zoom_ratio / 2)

#     # 확대 적용
#     plt.xlim(x_min_zoom, x_max_zoom)
#     plt.ylim(y_min_zoom, y_max_zoom)
    
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     output_path = os.path.join(save_folder, f"{base_name}_rotated.png")
#     plt.savefig(output_path, dpi=300)
#     plt.close()

    # print(f"회전 그래프 저장 완료: {output_path}")