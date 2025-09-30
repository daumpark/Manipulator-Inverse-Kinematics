# analytical_ik/solvers.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# -----------------------------
# 2-DOF Planar Analytical IK
# -----------------------------
@dataclass
class Planar2DParams:
    L1: float = 0.35
    L2: float = 0.25
    joint_limits: Tuple[Tuple[float,float], Tuple[float,float]] = ((-np.pi, np.pi), (-np.pi, np.pi))

class Planar2DAnalyticalIK:
    """
    2-DOF 평면 매니퓰레이터 (z축 회전 2개) 해석해 IK.
    목표는 (x,y) 위치만 사용. (z/orientation 무시)
    """
    def __init__(self, params: Planar2DParams = Planar2DParams()):
        self.p = params

    def solve(self, target_xy: np.ndarray, q_seed: Optional[np.ndarray]=None, return_both=False):
        x, y = float(target_xy[0]), float(target_xy[1])
        L1, L2 = self.p.L1, self.p.L2

        r2 = x*x + y*y
        c2 = (r2 - L1*L1 - L2*L2) / (2*L1*L2)
        if c2 < -1.0 - 1e-8 or c2 > 1.0 + 1e-8:
            return (None, False, {'reason': 'unreachable'})
        c2 = np.clip(c2, -1.0, 1.0)
        s2_candidates = [ np.sqrt(1-c2*c2), -np.sqrt(1-c2*c2) ]  # elbow up/down

        sols = []
        for s2 in s2_candidates:
            q2 = np.arctan2(s2, c2)
            k1 = L1 + L2*np.cos(q2)
            k2 = L2*np.sin(q2)
            q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
            # 정규화 및 제한
            q = np.array([np.arctan2(np.sin(q1), np.cos(q1)), np.arctan2(np.sin(q2), np.cos(q2))])
            if self._within_limits(q):
                sols.append(q)

        if not sols:
            return (None, False, {'reason': 'no_solution_within_limits'})

        if return_both:
            return (sols, True, {'solutions': len(sols)})

        # seed 기준 가장 가까운 해 선택
        if q_seed is None:
            q_seed = np.zeros(2)
        best = min(sols, key=lambda q: np.linalg.norm((q - q_seed + np.pi)%(2*np.pi) - np.pi))
        return (best, True, {'solutions': len(sols)})

    def fk_points(self, q: np.ndarray):
        """시각화/오차 확인용 (joint, ee 좌표)."""
        L1, L2 = self.p.L1, self.p.L2
        q1, q2 = q
        p0 = np.array([0.0, 0.0])
        p1 = np.array([L1*np.cos(q1), L1*np.sin(q1)])
        p2 = p1 + np.array([L2*np.cos(q1+q2), L2*np.sin(q1+q2)])
        return np.stack([p0, p1, p2], axis=0)

    def _within_limits(self, q):
        lo1, hi1 = self.p.joint_limits[0]
        lo2, hi2 = self.p.joint_limits[1]
        return (lo1 <= q[0] <= hi1) and (lo2 <= q[1] <= hi2)


# ---------------------------------
# PiPER Analytical IK (Python port)
# ---------------------------------
class PiperAnalyticalIK:
    """
    업로드해주신 C++ 해더의 구성/아이디어를 그대로 파이썬으로 옮겼습니다:
      - DH 파라미터(STANDARD/MODIFIED)
      - 손목 중심(wrist center) → (q1,q2,q3) (팔)
      - R36에서 ZYZ 형태로 (q4,q5,q6) (손목)
      - theta_offset 적용 후 [-pi,pi] 정규화 및 조인트 리밋 필터
    """
    STANDARD = 'standard'
    MODIFIED = 'modified'

    def __init__(self, dh_type: str = STANDARD):
        assert dh_type in (self.STANDARD, self.MODIFIED)
        self.dh_type = dh_type
        # PiperForwardKinematics.hpp의 DH 파라미터를 그대로 사용합니다.
        # [alpha, a, d, theta_offset]
        if dh_type == self.STANDARD:
            self.dh = np.array([
                [-np.pi/2,   0.0,       0.123,     0.0                     ],  # 1
                [0.0,        0.28503,   0.0,       -172.22/180*np.pi       ],  # 2
                [ np.pi/2,  -0.021984,  0.0,       -102.78/180*np.pi       ],  # 3
                [-np.pi/2,   0.0,       0.25075,   0.0                     ],  # 4
                [ np.pi/2,   0.0,       0.0,       0.0                     ],  # 5
                [ 0.0,       0.0,       0.211,     0.0                     ],  # 6
            ])
        else:
            self.dh = np.array([
                [ 0.0,       0.0,       0.123,     0.0                     ],  # 1
                [-np.pi/2,   0.0,       0.0,       -172.22/180*np.pi       ],  # 2
                [ 0.0,       0.28503,   0.0,       -102.78/180*np.pi       ],  # 3
                [ np.pi/2,  -0.021984,  0.25075,   0.0                     ],  # 4
                [-np.pi/2,   0.0,       0.0,       0.0                     ],  # 5
                [ np.pi/2,   0.0,       0.211,     0.0                     ],  # 6
            ])

    # ---------- Public API ----------
    def compute_ik(self, T_target: np.ndarray, joint_limits: Optional[List[Tuple[float,float]]]=None,
                   q_seed: Optional[np.ndarray]=None, filter_by_limits=True) -> List[np.ndarray]:
        p = T_target[:3, 3]
        R = T_target[:3, :3]
        p_wc = self._wrist_center(p, R)          # wrist center
        arm_sols = self._solve_arm(p_wc)         # (q1,q2,q3) 후보들
        all_solutions = []

        for q1, q2, q3 in arm_sols:
            R03 = self._R03(q1, q2, q3)          # offsets 포함한 R03
            R36 = R03.T @ R                      # 원하는 손목 회전
            for q4, q5, q6 in self._solve_wrist(R36):
                q = np.array([q1, q2, q3, q4, q5, q6], float)
                q = self._apply_offsets(q)       # theta_offset 제거하여 '조인트 각'으로 변환
                q = self._wrap(q)
                if (not filter_by_limits) or (joint_limits is None) or self._within_limits(q, joint_limits):
                    all_solutions.append(q)

        if not all_solutions:
            return []

        # seed와 가장 가까운 해 선택(우선순위) + 전체 반환
        if q_seed is None:
            q_seed = np.zeros(6)
        all_solutions.sort(key=lambda qq: self._ang_dist(qq, q_seed))
        return all_solutions

    # ---------- Internals ----------
    def _wrist_center(self, p_target, R_target):
        # C++ 헤더와 동일: p_wc = p - d6 * z6
        d6 = self.dh[5, 2]
        z6 = R_target[:, 2]
        return p_target - d6 * z6

    def _solve_arm(self, p_wc):
        x, y, z = float(p_wc[0]), float(p_wc[1]), float(p_wc[2])

        # Piper 헤더에서 사용한 등가 길이
        a2, a3 = self.dh[1,1], self.dh[2,1]
        d1, d4 = self.dh[0,2], self.dh[3,2]
        z_eff = z - d1
        r = np.hypot(x, y)
        L2 = a2
        L3 = np.sqrt(a3*a3 + d4*d4)

        D = (r*r + z_eff*z_eff - L2*L2 - L3*L3) / (2*L2*L3)
        if np.abs(D) > 1.0 + 1e-6:
            return []  # reach 불가
        D = np.clip(D, -1.0, 1.0)

        beta = np.arccos(D)
        phi  = np.arctan2(d4, np.abs(a3))   # a3<0 반영

        sols = []
        for sgn in (1.0, -1.0):
            q3 = sgn*beta - phi
            k1 = L2 + L3*np.cos(q3 + phi)
            k2 = L3*np.sin(q3 + phi)
            gamma = np.arctan2(z_eff, r)
            delta = np.arctan2(k2, k1)
            q2 = gamma - delta
            q1 = np.arctan2(y, x)
            sols.append((q1, q2, q3))
        return sols

    def _R03(self, q1, q2, q3):
        al = self.dh[:3, 0]; a = self.dh[:3, 1]; d = self.dh[:3, 2]; off = self.dh[:3, 3]
        T01 = self._dh_T(al[0], a[0], d[0], q1 + off[0])
        T12 = self._dh_T(al[1], a[1], d[1], q2 + off[1])
        T23 = self._dh_T(al[2], a[2], d[2], q3 + off[2])
        return (T01 @ T12 @ T23)[:3, :3]

    def _solve_wrist(self, R36):
        # ZYZ 스타일 해 (C++ 헤더 구현과 동일한 형태)
        r11, r12, r13 = R36[0,0], R36[0,1], R36[0,2]
        r21, r22, r23 = R36[1,0], R36[1,1], R36[1,2]
        r31, r32, r33 = R36[2,0], R36[2,1], R36[2,2]

        sols = []
        for sgn in (1.0, -1.0):
            if abs(r33) > 0.9999:
                q5 = 0.0 if (r33 > 0) else np.pi
                q4 = 0.0
                q6 = np.arctan2(r21, r11)
            else:
                q5 = sgn * np.arccos(np.clip(r33, -1.0, 1.0))
                q4 = np.arctan2(r23, r13)
                q6 = np.arctan2(r32, -r31)
                if sgn < 0:
                    q4 += np.pi
                    q6 -= np.pi
                q4 = np.arctan2(np.sin(q4), np.cos(q4))
                q6 = np.arctan2(np.sin(q6), np.cos(q6))
            sols.append((q4, q5, q6))
        return sols

    def _dh_T(self, alpha, a, d, theta):
        # PiperForwardKinematics.hpp의 computeTransform와 동일 수식
        if self.dh_type == self.STANDARD:
            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(theta), np.sin(theta)
            return np.array([
                [ ct, -st*ca,  st*sa, a*ct],
                [ st,  ct*ca, -ct*sa, a*st],
                [  0,     sa,     ca,   d ],
                [  0,      0,      0,   1 ],
            ])
        else:  # MODIFIED
            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(theta), np.sin(theta)
            return np.array([
                [ ct,          -st,           0,     a],
                [ st*ca,  ct*ca,        -sa, -sa*d],
                [ st*sa,  ct*sa,         ca,  ca*d],
                [ 0,           0,           0,     1],
            ])

    def _apply_offsets(self, q):
        # 최종적으로 조인트 명령각 = theta - offset
        off = self.dh[:, 3]
        return q - off

    @staticmethod
    def _wrap(q):
        return (q + np.pi)%(2*np.pi) - np.pi

    @staticmethod
    def _ang_dist(a, b):
        d = (a - b + np.pi)%(2*np.pi) - np.pi
        return float(np.linalg.norm(d))

    @staticmethod
    def _within_limits(q, limits):
        for i, (lo, hi) in enumerate(limits):
            if not (lo-1e-9 <= q[i] <= hi+1e-9):
                return False
        return True
