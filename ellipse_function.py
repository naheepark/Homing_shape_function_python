import numpy as np
import matplotlib.pyplot as plt
import random

# 타원 파라미터
h, k = 0, 0     # 중심 좌표
a, b = 5, 3     # 장축, 단축 반지름

# 중간 2/3 범위의 x좌표 구간
x_min = h - (2 * a / 3)
x_max = h + (2 * a / 3)

# x 좌표 범위에서 충분히 많은 x 값 생성
x_vals = np.linspace(x_min, x_max, 1000)

# 위쪽 타원 곡선의 y 값 계산
y_vals = k + b * np.sqrt(1 - ((x_vals - h) ** 2) / a**2)

# (x, y) 쌍으로 묶기
curve_points = list(zip(x_vals, y_vals))

# 임의로 10개 샘플 추출
sampled_points = random.sample(curve_points, 10)

# 결과 출력
print("샘플링된 타원 곡선 위 점 10개:")
for pt in sampled_points:
    print(f"({pt[0]:.4f}, {pt[1]:.4f})")

# 시각화
plt.figure(figsize=(6,6))
plt.plot(x_vals, y_vals, label='Upper Ellipse Curve', color='blue')
x_sample = [p[0] for p in sampled_points]
y_sample = [p[1] for p in sampled_points]
plt.scatter(x_sample, y_sample, color='red', label='Sampled 10 Points', zorder=5)
plt.axvline(x_min, color='gray', linestyle='--', linewidth=1)
plt.axvline(x_max, color='gray', linestyle='--', linewidth=1)
plt.axvline(h, color='black', linestyle='--', linewidth=1, label='Center')
plt.title("Ellipse 2/3 Upper Segment")
plt.legend()
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()
