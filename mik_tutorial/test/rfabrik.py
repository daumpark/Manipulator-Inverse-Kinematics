# -*- coding: utf-8 -*-
"""
FABRIK-R 스타일 시각화 (Schilling Titan 2)
- 논문 표 I의 DH 파라미터 사용
- 2~4번 관절 공동평면 Φ_{2,3,4}, 5번 관절의 보조 평면 Ω를 이용한 전/후진 반복
- 교육/시각화용 간소화 구현
- [FIX] 인덱스/앵커링: p0(월드 원점) 고정, p1은 조인트1의 DH 파라미터로부터 매 반복 강제 설정
"""

import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --------------------------
# Titan2 DH 파라미터 (논문 표 I; m/deg)
# --------------------------
T2_d  = np.array([0.123, 0, 0, 0.25075, 0, 0.091], dtype=float)
T2_a  = np.array([0, 0, 0.28503, -0.02198, 0, 0], dtype=float)
T2_alpha_deg = np.array([0, -90, 0, 90, -90, 90], dtype=float)
T2_alpha = np.deg2rad(T2_alpha_deg)

# --------------------------
# 유틸/수학 함수
# --------------------------
def dh_transform(theta, d, a, alpha):
    """표준 DH 변환행렬 (θ: z회전, d: z이동, a: x이동, α: x회전)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,     sa,     ca,     d],
        [0.,    0.,     0.,     1.]
    ])

def forward_kinematics_titan2(thetas):
    """
    Titan2의 DH 파라미터로 p0..p6(7개 점) 좌표 반환.
    thetas: 6개 조인트 각도(rad)
    """
    T = np.eye(4)
    pts = [T[:3,3].copy()]  # p0 (base at world origin)
    for i in range(6):
        T = T @ dh_transform(thetas[i], T2_d[i], T2_a[i], T2_alpha[i])
        pts.append(T[:3,3].copy())
    return np.array(pts)  # (7,3)

def segment_lengths_from_points(points):
    """연속 점 사이 거리(세그먼트 길이) 6개."""
    return np.linalg.norm(points[1:] - points[:-1], axis=1)

def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n

def project_point_to_plane(p, n, p0):
    """법선 n, 기준점 p0로 정의된 평면에 점 p를 직교사영."""
    n = normalize(n)
    return p - np.dot(p - p0, n) * n

def fabrik_place(prev, curr, length):
    """
    FABRIK 한 스텝: prev(고정)에서 curr 방향으로 length만큼 떨어진 지점.
    (curr는 원하는 방향성 제공용)
    """
    d = np.linalg.norm(curr - prev)
    if d < 1e-12:
        return prev.copy()
    return prev + (curr - prev) * (length / d)

def make_vertical_plane_through(base_point, through_point):
    """
    베이스점과 임의의 점을 지나는 '수직(=법선이 XY평면에만 존재)' 평면 생성.
    반환: (법선벡터, 평면상의 점)
    """
    z = np.array([0.0, 0.0, 1.0])
    v = through_point - base_point
    v_xy = np.array([v[0], v[1], 0.0])
    if np.linalg.norm(v_xy) < 1e-9:
        normal = np.array([1.0, 0.0, 0.0])  # 바로 위라면 YZ평면
    else:
        normal = np.cross(z, v_xy)
    return normalize(normal), base_point.copy()

def draw_plane(ax, normal, point, size, alpha_val=0.18, color=None):
    """시각화를 위한 직사각형 평면 패치."""
    n = normalize(normal)
    # 평면을 생성할 두 기저 벡터
    a = np.array([1.0, 0.0, 0.0]) if abs(np.dot(n, [1,0,0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(n, a))
    v = normalize(np.cross(n, u))
    s = size
    corners = np.array([
        point + s*( u + v),
        point + s*( u - v),
        point + s*(-u - v),
        point + s*(-u + v)
    ])
    poly = Poly3DCollection([corners], alpha=alpha_val, facecolor=color)
    ax.add_collection3d(poly)

# --------------------------
# [핵심 FIX] p1을 DH로부터 강제 앵커링
# --------------------------
def joint1_pose_from_dh(base_origin, theta1):
    """
    조인트1(DH의 0번째 행)로부터 p1의 '월드' 위치를 계산.
    Titan2 표 I에서는 a0=0, alpha0=0 → p1 = p0 + [0,0,d0] (theta1와 무관)
    일반성 위해 DH로 계산한 뒤 base_origin을 더해준다.
    """
    T1 = dh_transform(theta1, T2_d[0], T2_a[0], T2_alpha[0])
    return base_origin + T1[:3, 3]

# --------------------------
# FABRIK-R (간소화) 반복
# --------------------------
def fabrik_r_iterate(points, lengths, target, base_origin, thetas_for_dh, max_iter=60, tol=1e-3):
    """
    points: 초기 p0..p6 (p0=base 고정)
    lengths: 6개 세그먼트 길이
    target: p6 목표
    base_origin: 월드 좌표계 베이스(p0)
    thetas_for_dh: DH 계산용 각도(특히 theta1). Titan2에선 p1 위치는 theta1와 무관하나 일반성 유지용.
    """
    pts = points.copy()

    # --- 고정 앵커 ---
    p0 = base_origin.copy()
    p1_anchor = joint1_pose_from_dh(p0, thetas_for_dh[0])

    # 초기 고정 상태 반영
    pts[0] = p0
    pts[1] = p1_anchor

    history = [pts.copy()]

    # 초기 Φ 평면(시각화용)
    phi_normal, phi_point = make_vertical_plane_through(p0, pts[5])
    omega_n, omega_p = phi_normal.copy(), phi_point.copy()

    for it in range(max_iter):
        # ----- 고정점 재적용 (전진 단계 전에 p0/p1을 확실히 고정) -----
        print(it)
        pts[0] = p0
        pts[1] = p1_anchor

        # ---------- Forward ----------
        pts[6] = target.copy()                     # 엔드 이펙터를 목표로
        pts[5] = fabrik_place(pts[6], pts[5], lengths[5])

        # Ω: 베이스와 p5'을 지나는 수직 평면
        omega_n, omega_p = make_vertical_plane_through(p0, pts[5])

        # p4, p3, p2를 Ω에 두고 당김 (p1은 고정이므로 업데이트 금지)
        for i in [4, 3, 2]:
            proj = project_point_to_plane(pts[i], omega_n, omega_p)
            pts[i] = fabrik_place(pts[i+1], proj, lengths[i])

        # ---------- Backward ----------
        # 고정 앵커 재설정
        pts[0] = p0
        pts[1] = p1_anchor

        # Φ: 베이스와 p5'을 포함하는 수직평면(2–4 공면)
        phi_normal, phi_point = make_vertical_plane_through(p0, pts[5])

        # p2..p6: FABRIK 전진, 단 p2,p3,p4는 Φ에 사영
        for i in [1, 2, 3, 4, 5]:  # i는 세그먼트 인덱스 (p_i → p_{i+1})
            prev = pts[i]          # p1은 이미 고정
            nxt = pts[i+1]
            if i+1 in (2, 3, 4):
                nxt = project_point_to_plane(nxt, phi_normal, phi_point)
            pts[i+1] = fabrik_place(prev, nxt, lengths[i])

        history.append(pts.copy())
        if np.linalg.norm(pts[6] - target) < tol:
            break

    return pts, history, phi_normal, phi_point, omega_n, omega_p

# --------------------------
# 데모 실행
# --------------------------
if __name__ == "__main__":
    # 초기 각도 (rad)
    thetas0 = np.zeros(6)

    # p0..p6 초기 위치 및 길이
    points0 = forward_kinematics_titan2(thetas0)     # p0..p6
    lengths  = segment_lengths_from_points(points0)  # 세그먼트 길이 고정

    # 타깃(도달 가능 범위 내로 설정)
    target = np.array([0.3, 0.1, 0.3])  # [m]

    # FABRIK-R 반복 (p0=월드 원점 고정, p1=DH로 앵커)
    sol, hist, phi_n, phi_p, omega_n, omega_p = fabrik_r_iterate(
        points0, lengths, target, base_origin=points0[0], thetas_for_dh=thetas0, max_iter=100, tol=1e-3
    )

    # ---------------- Visualization ----------------
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # # 초기 체인
    # p_init = hist[0]
    # ax.plot(p_init[:,0], p_init[:,1], p_init[:,2], marker='o', linewidth=2, label='initial')

    # 결과 체인
    ax.plot(sol[:,0], sol[:,1], sol[:,2], marker='o', linewidth=3, label='FABRIK-R result')

    # 타깃
    ax.scatter([target[0]], [target[1]], [target[2]], s=70, label='target')

    # 평면 표시 (최종 Φ, 마지막 Ω)
    span = np.max(np.linalg.norm(sol - sol[0], axis=1))
    draw_plane(ax, phi_n,  phi_p,  size=span*0.7)   # Φ_{2,3,4}
    draw_plane(ax, omega_n, omega_p, size=span*0.6) # Ω

    # 라벨
    for i in range(7):
        ax.text(sol[i,0], sol[i,1], sol[i,2], f"p{i}", fontsize=9)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("FABRIK-R style solution (Titan 2) — p0 fixed, p1 anchored by DH")
    ax.legend(loc='upper left')

    # 보기 좋게 등축비
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = 0.6 * np.max(limits[:,1] - limits[:,0])
    ax.set_xlim3d([center[0]-radius, center[0]+radius])
    ax.set_ylim3d([center[1]-radius, center[1]+radius])
    ax.set_zlim3d([center[2]-radius, center[2]+radius])

    plt.show()
