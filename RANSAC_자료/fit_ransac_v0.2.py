import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class ProfileInterpolator:
    def __init__(self, ransac_threshold=2.0, min_samples=10):
        """
        프로파일 보간 클래스

        Args:
            ransac_threshold: RANSAC outlier threshold
            min_samples: 최소 샘플 수
        """
        self.ransac_threshold = ransac_threshold
        self.min_samples = min_samples
        self.segments = []

    def load_data(self, filename):
        """데이터 파일 로드"""
        data = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    data.append([x, y])
        return np.array(data)

    def fit_polynomial_ransac(self, points, degree=3):
        """RANSAC을 이용한 다항식 피팅 (원호 피팅 실패시 대안)"""
        if len(points) < degree + 1:
            return None

        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        # 다항식 특성 생성
        poly_features = PolynomialFeatures(degree=degree)

        # RANSAC 회귀
        ransac = RANSACRegressor(
            estimator=Pipeline([
                ('poly', poly_features),
                ('linear', RANSACRegressor())
            ]),
            residual_threshold=self.ransac_threshold,
            min_samples=max(degree + 1, len(points) // 10),
            max_trials=1000,
            random_state=42
        )

        try:
            # 간단한 다항식 피팅으로 대체
            coeffs = np.polyfit(points[:, 0], points[:, 1], degree)
            return {
                'type': 'polynomial',
                'coefficients': coeffs,
                'degree': degree
            }
        except:
            return None

    def fit_line_ransac(self, points):
        """RANSAC을 이용한 직선 피팅"""
        if len(points) < 2:
            return None

        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]

        ransac = RANSACRegressor(
            residual_threshold=self.ransac_threshold,
            min_samples=max(2, len(points) // 10),
            max_trials=1000,
            random_state=42
        )

        try:
            ransac.fit(X, y)
            return {
                'type': 'line',
                'slope': ransac.estimator_.coef_[0],
                'intercept': ransac.estimator_.intercept_,
                'inliers': ransac.inlier_mask_,
                'score': ransac.score(X, y)
            }
        except:
            return None

    def fit_ellipse_ransac(self, points):
        """RANSAC을 이용한 타원 피팅"""
        if len(points) < 5:
            return None

        def ellipse_residuals(params, points):
            """타원 방정식: (x-h)²/a² + (y-k)²/b² = 1"""
            h, k, a, b, theta = params

            # 회전 변환
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x_rot = (points[:, 0] - h) * cos_theta + (points[:, 1] - k) * sin_theta
            y_rot = -(points[:, 0] - h) * sin_theta + (points[:, 1] - k) * cos_theta

            # 타원 방정식 계산
            ellipse_eq = (x_rot**2)/(a**2) + (y_rot**2)/(b**2)
            return np.abs(ellipse_eq - 1)

        best_params = None
        best_inliers = None
        best_score = float('inf')
        max_iterations = 2000

        for _ in range(max_iterations):
            if len(points) < 5:
                break

            # 랜덤하게 5개 점 선택
            sample_indices = np.random.choice(len(points), 5, replace=False)
            sample_points = points[sample_indices]

            try:
                # 초기 타원 파라미터 추정
                ellipse_params = self.fit_ellipse_5points(sample_points)
                if ellipse_params is None:
                    continue

                # 모든 점에 대한 잔차 계산
                residuals = ellipse_residuals(ellipse_params, points)
                inliers = residuals < (self.ransac_threshold / 10)  # 타원은 더 엄격한 기준

                if np.sum(inliers) > max(5, len(points) * 0.3):
                    inlier_points = points[inliers]
                    refined_params = self.refine_ellipse(inlier_points, ellipse_params)

                    if refined_params is not None:
                        refined_residuals = ellipse_residuals(refined_params, points)
                        inliers_refined = refined_residuals < (self.ransac_threshold / 10)
                        score = np.mean(refined_residuals[inliers_refined])

                        if score < best_score and np.sum(inliers_refined) > len(points) * 0.3:
                            best_score = score
                            best_params = refined_params
                            best_inliers = inliers_refined
            except:
                continue

        if best_params is not None:
            return {
                'type': 'ellipse',
                'center': (best_params[0], best_params[1]),
                'axes': (best_params[2], best_params[3]),
                'angle': best_params[4],
                'inliers': best_inliers,
                'score': best_score
            }
        return None

    def fit_ellipse_5points(self, points):
        """5점으로 타원 피팅"""
        try:
            # 간단한 초기 추정
            x_center = np.mean(points[:, 0])
            y_center = np.mean(points[:, 1])

            # 중심으로부터의 거리들을 이용해 축 길이 추정
            distances = np.sqrt((points[:, 0] - x_center)**2 + (points[:, 1] - y_center)**2)
            a = np.max(distances) * 1.2
            b = np.min(distances) * 0.8

            if b <= 0:
                b = a * 0.5

            theta = 0  # 초기 회전각

            return [x_center, y_center, a, b, theta]
        except:
            return None

    def refine_ellipse(self, points, initial_params):
        """최소제곱법으로 타원 파라미터 정제"""
        try:
            def objective(params):
                h, k, a, b, theta = params
                if a <= 0 or b <= 0:
                    return 1e10

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                x_rot = (points[:, 0] - h) * cos_theta + (points[:, 1] - k) * sin_theta
                y_rot = -(points[:, 0] - h) * sin_theta + (points[:, 1] - k) * cos_theta

                ellipse_eq = (x_rot**2)/(a**2) + (y_rot**2)/(b**2)
                return np.sum((ellipse_eq - 1)**2)

            # 파라미터 범위 제한
            bounds = [
                (initial_params[0] - 50, initial_params[0] + 50),  # h
                (initial_params[1] - 50, initial_params[1] + 50),  # k
                (max(1, initial_params[2] * 0.5), initial_params[2] * 2),  # a
                (max(1, initial_params[3] * 0.5), initial_params[3] * 2),  # b
                (-np.pi/2, np.pi/2)  # theta
            ]

            result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
            if result.success:
                return result.x
            return initial_params
        except:
            return initial_params

    def process_profile_custom(self, points):
        """4개 구간으로 분할: 선분1-선분2-원호-선분3 (부드럽게 연결)"""
        self.segments = []
        total_points = len(points)

        # 4개 구간 경계 설정
        segment_boundaries = [
            0,           # 시작
            220,         # 첫 번째 선분 끝
            350,         # 두 번째 선분 끝
            570,         # 원호 끝
            total_points # 마지막 선분 끝
        ]

        print(f"Processing {total_points} points with 4-segment division:")
        print(f"  Segment 1: Line 1 (index 0 to {segment_boundaries[1]})")
        print(f"  Segment 2: Line 2 (index {segment_boundaries[1]} to {segment_boundaries[2]})")
        print(f"  Segment 3: Arc (index {segment_boundaries[2]} to {segment_boundaries[3]})")
        print(f"  Segment 4: Line 3 (index {segment_boundaries[3]} to {segment_boundaries[4]})")

        # 1. 첫 번째 선분 구간 (0 ~ 227)
        line1_points = points[segment_boundaries[0]:segment_boundaries[1]+1]  # 227 포함
        if len(line1_points) > 5:
            line_result = self.fit_line_ransac(line1_points)
            if line_result:
                self.segments.append({
                    'type': 'line',
                    'points': line1_points,
                    'params': line_result,
                    'start_idx': segment_boundaries[0],
                    'end_idx': segment_boundaries[1],
                    'name': 'Line 1 (First Descent)'
                })
                print(f"    Line 1: {len(line1_points)} points, slope={line_result['slope']:.4f}")

        # 2. 두 번째 선분 구간 (228 ~ 340) - 연결점에서 부드럽게 이어지도록 조정
        line2_points = points[segment_boundaries[1]:segment_boundaries[2]+1]  # 227부터 340까지 (겹치는 점 포함)
        if len(line2_points) > 5:
            # 기본 선형 피팅
            line_result = self.fit_line_ransac(line2_points)
            if line_result:
                # 연결점에서의 연속성 확보
                if len(self.segments) > 0:
                    prev_seg = self.segments[-1]
                    connection_x = points[segment_boundaries[1], 0]  # 연결점 x 좌표
                    prev_y = prev_seg['params']['slope'] * connection_x + prev_seg['params']['intercept']

                    # 새로운 절편 계산 (연결점을 지나도록)
                    new_intercept = prev_y - line_result['slope'] * connection_x
                    line_result['intercept'] = new_intercept

                    print(f"    Line 2: Connected smoothly at x={connection_x:.1f}, y={prev_y:.1f}")

                self.segments.append({
                    'type': 'line',
                    'points': line2_points,
                    'params': line_result,
                    'start_idx': segment_boundaries[1],
                    'end_idx': segment_boundaries[2],
                    'name': 'Line 2 (Second Descent)'
                })
                print(f"    Line 2: {len(line2_points)} points, slope={line_result['slope']:.4f}")

        # 3. 원호 구간 (340 ~ 560) - 연결점에서 부드럽게 이어지도록 조정
        arc_points = points[segment_boundaries[2]:segment_boundaries[3]+1]  # 340부터 560까지
        if len(arc_points) > 10:
            # 연결점 정보 저장
            connection_info = None
            if len(self.segments) > 0:
                prev_seg = self.segments[-1]
                connection_x = points[segment_boundaries[2], 0]
                connection_y = prev_seg['params']['slope'] * connection_x + prev_seg['params']['intercept']
                connection_info = (connection_x, connection_y)

            # 타원 피팅 시도
            ellipse_result = self.fit_ellipse_ransac(arc_points)
            circle_result = self.fit_circle_ransac(arc_points)

            # 결과 비교하여 더 좋은 것 선택
            ellipse_valid = ellipse_result and np.sum(ellipse_result['inliers']) > len(arc_points) * 0.4
            circle_valid = circle_result and np.sum(circle_result['inliers']) > len(arc_points) * 0.4

            if ellipse_valid and circle_valid:
                ellipse_ratio = np.sum(ellipse_result['inliers']) / len(arc_points)
                circle_ratio = np.sum(circle_result['inliers']) / len(arc_points)

                if ellipse_ratio > circle_ratio * 1.1:
                    selected_result = ellipse_result
                    arc_type = 'ellipse'
                else:
                    selected_result = circle_result
                    arc_type = 'circle'
            elif ellipse_valid:
                selected_result = ellipse_result
                arc_type = 'ellipse'
            elif circle_valid:
                selected_result = circle_result
                arc_type = 'circle'
            else:
                selected_result = None
                arc_type = 'polynomial'

            if selected_result:
                self.segments.append({
                    'type': arc_type,
                    'points': arc_points,
                    'params': selected_result,
                    'start_idx': segment_boundaries[2],
                    'end_idx': segment_boundaries[3],
                    'name': f'Arc ({arc_type.capitalize()})',
                    'connection_info': connection_info
                })

                if arc_type == 'ellipse':
                    center = selected_result['center']
                    axes = selected_result['axes']
                    angle = selected_result['angle']
                    print(f"    Arc: Ellipse with {len(arc_points)} points")
                    print(f"          center=({center[0]:.1f}, {center[1]:.1f}), axes=({axes[0]:.1f}, {axes[1]:.1f}), angle={np.degrees(angle):.1f}°")
                else:
                    center = selected_result['center']
                    radius = selected_result['radius']
                    print(f"    Arc: Circle with {len(arc_points)} points")
                    print(f"          center=({center[0]:.1f}, {center[1]:.1f}), radius={radius:.1f}")
            else:
                # 다항식 피팅으로 대체
                poly_result = self.fit_polynomial_ransac(arc_points, degree=3)
                if poly_result:
                    self.segments.append({
                        'type': 'polynomial',
                        'points': arc_points,
                        'params': poly_result,
                        'start_idx': segment_boundaries[2],
                        'end_idx': segment_boundaries[3],
                        'name': 'Arc (Polynomial)',
                        'connection_info': connection_info
                    })
                    print(f"    Arc: Using polynomial fitting with {len(arc_points)} points")

        # 4. 마지막 선분 구간 (560 ~ 끝) - 연결점에서 부드럽게 이어지도록 조정
        line3_points = points[segment_boundaries[3]:segment_boundaries[4]]  # 560부터 끝까지
        if len(line3_points) > 5:
            line_result = self.fit_line_ransac(line3_points)
            if line_result:
                # 연결점에서의 연속성 확보
                if len(self.segments) > 0:
                    prev_seg = self.segments[-1]
                    connection_x = points[segment_boundaries[3], 0]

                    # 이전 세그먼트가 원호인 경우 연결점 계산
                    if prev_seg['type'] in ['circle', 'ellipse', 'polynomial']:
                        prev_y = self.evaluate_segment(prev_seg, connection_x)
                        # 새로운 절편 계산
                        new_intercept = prev_y - line_result['slope'] * connection_x
                        line_result['intercept'] = new_intercept
                        print(f"    Line 3: Connected smoothly at x={connection_x:.1f}, y={prev_y:.1f}")

                self.segments.append({
                    'type': 'line',
                    'points': line3_points,
                    'params': line_result,
                    'start_idx': segment_boundaries[3],
                    'end_idx': segment_boundaries[4],
                    'name': 'Line 3 (Ascent)'
                })
                print(f"    Line 3: {len(line3_points)} points, slope={line_result['slope']:.4f}")

        return self.segments

    def fit_circle_ransac(self, points):
        """RANSAC을 이용한 원 피팅"""
        if len(points) < 3:
            return None

        def circle_residuals(params, points):
            cx, cy, r = params
            distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
            return np.abs(distances - r)

        best_params = None
        best_inliers = None
        best_score = float('inf')
        max_iterations = 1000

        for _ in range(max_iterations):
            if len(points) < 3:
                break

            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]

            try:
                circle_params = self.fit_circle_3points(sample_points)
                if circle_params is None:
                    continue

                cx, cy, r = circle_params

                residuals = circle_residuals([cx, cy, r], points)
                inliers = residuals < self.ransac_threshold

                if np.sum(inliers) > max(3, len(points) * 0.2):
                    inlier_points = points[inliers]
                    refined_params = self.refine_circle(inlier_points, [cx, cy, r])

                    if refined_params is not None:
                        refined_residuals = circle_residuals(refined_params, points)
                        inliers_refined = refined_residuals < self.ransac_threshold
                        score = np.mean(refined_residuals[inliers_refined])

                        if score < best_score:
                            best_score = score
                            best_params = refined_params
                            best_inliers = inliers_refined
            except:
                continue

        if best_params is not None:
            return {
                'type': 'circle',
                'center': (best_params[0], best_params[1]),
                'radius': best_params[2],
                'inliers': best_inliers,
                'score': best_score
            }
        return None

    def fit_circle_3points(self, points):
        """3점으로 원의 중심과 반지름 계산"""
        try:
            p1, p2, p3 = points
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3

            det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if abs(det) < 1e-10:
                return None

            a = x1**2 + y1**2
            b = x2**2 + y2**2
            c = x3**2 + y3**2

            cx = (a*(y2-y3) + b*(y3-y1) + c*(y1-y2)) / (2*det)
            cy = (a*(x3-x2) + b*(x1-x3) + c*(x2-x1)) / (2*det)

            r = np.sqrt((cx-x1)**2 + (cy-y1)**2)

            return [cx, cy, r]
        except:
            return None

    def refine_circle(self, points, initial_params):
        """최소제곱법으로 원 파라미터 정제"""
        try:
            def objective(params):
                cx, cy, r = params
                distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
                return np.sum((distances - r)**2)

            result = minimize(objective, initial_params, method='L-BFGS-B')
            if result.success:
                return result.x
            return initial_params
        except:
            return initial_params

    def generate_continuous_profile(self, x_range=None, num_points=2000):
        """연속적인 프로파일 생성 (선형 보간으로 부드럽게 연결)"""
        if not self.segments:
            return None, None

        if x_range is None:
            all_points = np.vstack([seg['points'] for seg in self.segments])
            x_range = (np.min(all_points[:, 0]), np.max(all_points[:, 0]))

        x_smooth = np.linspace(x_range[0], x_range[1], num_points)
        y_smooth = np.zeros_like(x_smooth)

        # 각 세그먼트의 연결점 찾기
        connection_points = []
        for i in range(len(self.segments) - 1):
            seg1 = self.segments[i]
            seg2 = self.segments[i + 1]

            # 세그먼트 경계에서의 y 값 계산
            boundary_x = seg1['points'][-1, 0]  # 첫 번째 세그먼트의 마지막 x

            y1 = self.evaluate_segment(seg1, boundary_x)
            y2 = self.evaluate_segment(seg2, boundary_x)

            connection_points.append((boundary_x, y1, y2))

        # 각 x에 대해 해당하는 세그먼트의 y 값 계산
        for i, x in enumerate(x_smooth):
            for j, seg in enumerate(self.segments):
                seg_points = seg['points']
                x_min, x_max = np.min(seg_points[:, 0]), np.max(seg_points[:, 0])

                if x_min <= x <= x_max:
                    y_smooth[i] = self.evaluate_segment(seg, x)
                    break
            else:
                # 세그먼트 범위를 벗어난 경우 가장 가까운 세그먼트 사용
                distances = []
                for seg in self.segments:
                    seg_points = seg['points']
                    x_min, x_max = np.min(seg_points[:, 0]), np.max(seg_points[:, 0])
                    if x < x_min:
                        distances.append(x_min - x)
                    elif x > x_max:
                        distances.append(x - x_max)
                    else:
                        distances.append(0)

                closest_seg = self.segments[np.argmin(distances)]
                y_smooth[i] = self.evaluate_segment(closest_seg, x)

        return x_smooth, y_smooth

    def evaluate_segment(self, seg, x):
        """세그먼트에서 x 위치의 y 값 계산"""
        if seg['type'] == 'line':
            params = seg['params']
            return params['slope'] * x + params['intercept']

        elif seg['type'] == 'polynomial':
            params = seg['params']
            coeffs = params['coefficients']
            return np.polyval(coeffs, x)

        elif seg['type'] == 'circle':
            params = seg['params']
            cx, cy = params['center']
            r = params['radius']

            dx = x - cx
            if abs(dx) <= r:
                dy = np.sqrt(r**2 - dx**2)
                # 원래 데이터의 평균 y 위치를 기준으로 위/아래 결정
                avg_y = np.mean(seg['points'][:, 1])
                if avg_y > cy:
                    return cy + dy
                else:
                    return cy - dy
            else:
                # 원 범위를 벗어나면 가장 가까운 점 사용
                distances = np.abs(seg['points'][:, 0] - x)
                closest_idx = np.argmin(distances)
                return seg['points'][closest_idx, 1]

        elif seg['type'] == 'ellipse':
            params = seg['params']
            h, k = params['center']
            a, b = params['axes']
            theta = params['angle']

            # 회전된 타원에서 y 값 계산
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # x에서 타원의 y 값을 구하기 위해 방정식 풀이
            # (x-h)cosθ + (y-k)sinθ = X'
            # -(x-h)sinθ + (y-k)cosθ = Y'
            # X'^2/a^2 + Y'^2/b^2 = 1

            x_shifted = x - h

            # 이차방정식을 풀어서 y 값 구하기
            A = (sin_theta**2)/(a**2) + (cos_theta**2)/(b**2)
            B = 2 * x_shifted * (sin_theta * cos_theta * (1/(a**2) - 1/(b**2)))
            C = (x_shifted**2) * ((cos_theta**2)/(a**2) + (sin_theta**2)/(b**2)) - 1

            discriminant = B**2 - 4*A*C
            if discriminant >= 0:
                y1 = (-B + np.sqrt(discriminant)) / (2*A) + k
                y2 = (-B - np.sqrt(discriminant)) / (2*A) + k

                # 원래 데이터에 더 가까운 y 값 선택
                avg_y = np.mean(seg['points'][:, 1])
                if abs(y1 - avg_y) < abs(y2 - avg_y):
                    return y1
                else:
                    return y2
            else:
                # 타원 범위를 벗어나면 가장 가까운 점 사용
                distances = np.abs(seg['points'][:, 0] - x)
                closest_idx = np.argmin(distances)
                return seg['points'][closest_idx, 1]

        # 기본값
        return 0

    def visualize_results(self, original_points, show_segments=True):
        """결과 시각화"""
        plt.figure(figsize=(16, 10))

        # 원본 데이터와 세그먼트
        plt.subplot(2, 1, 1)
        plt.plot(original_points[:, 0], original_points[:, 1], 'b.', alpha=0.4, label='Original Data', markersize=2)

        if show_segments:
            colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']
            for i, seg in enumerate(self.segments):
                color = colors[i % len(colors)]
                seg_points = seg['points']
                plt.plot(seg_points[:, 0], seg_points[:, 1], 'o', color=color,
                        label=f"{seg['name']}", markersize=3, alpha=0.7)

                # 피팅된 곡선 그리기
                x_seg = np.linspace(np.min(seg_points[:, 0]), np.max(seg_points[:, 0]), 200)
                y_seg = []

                for x in x_seg:
                    y_seg.append(self.evaluate_segment(seg, x))

                plt.plot(x_seg, y_seg, '-', color=color, linewidth=3, alpha=0.8)

        plt.title('4-Segment Profile: Line1-Line2-Arc-Line3 (Smooth Connection)', fontsize=14)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # 연속적인 프로파일
        plt.subplot(2, 1, 2)
        plt.plot(original_points[:, 0], original_points[:, 1], 'b.', alpha=0.3, label='Original Data', markersize=1)

        x_smooth, y_smooth = self.generate_continuous_profile()
        if x_smooth is not None:
            plt.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='Continuous Profile (Line1-Line2-Arc-Line3)', alpha=0.9)

        plt.title('Final Continuous Profile', fontsize=14)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# 사용 예제
def main():
    # 프로파일 보간기 생성
    interpolator = ProfileInterpolator(ransac_threshold=4.0, min_samples=15)

    # 데이터 로드
    try:
        points = interpolator.load_data('coordinates.txt')
        print(f"Loaded {len(points)} data points")

        # 사용자 정의 프로파일 처리
        segments = interpolator.process_profile_custom(points)
        print(f"\nDetected {len(segments)} segments in Line1-Line2-Arc-Line3 pattern:")

        for i, seg in enumerate(segments):
            seg_type = seg['type']
            num_points = len(seg['points'])
            print(f"  {seg['name']}: {seg_type} ({num_points} points)")

            if seg_type == 'line':
                slope = seg['params']['slope']
                intercept = seg['params']['intercept']
                print(f"    -> slope={slope:.4f}, intercept={intercept:.2f}")
            elif seg_type == 'circle':
                center = seg['params']['center']
                radius = seg['params']['radius']
                print(f"    -> center=({center[0]:.1f}, {center[1]:.1f}), radius={radius:.1f}")
            elif seg_type == 'ellipse':
                center = seg['params']['center']
                axes = seg['params']['axes']
                angle = seg['params']['angle']
                print(f"    -> center=({center[0]:.1f}, {center[1]:.1f}), axes=({axes[0]:.1f}, {axes[1]:.1f}), angle={np.degrees(angle):.1f}°")

        # 결과 시각화
        interpolator.visualize_results(points, show_segments=True)

        # 연속 프로파일 데이터 저장
        x_smooth, y_smooth = interpolator.generate_continuous_profile(num_points=3000)
        if x_smooth is not None:
            smooth_data = np.column_stack([x_smooth, y_smooth])
            np.savetxt('continuous_profile.txt', smooth_data, fmt='%.6f', delimiter='\t')
            print(f"\nContinuous profile saved to 'continuous_profile.txt'")
            print(f"Profile consists of {len(segments)} segments: {' -> '.join([seg['name'] for seg in segments])}")

    except FileNotFoundError:
        print("coordinates.txt 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
