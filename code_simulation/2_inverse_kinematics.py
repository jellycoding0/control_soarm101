import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

soarm_dh_params = [
    {'alpha': 0.0,       'a': 0.000, 'd': 0.119, 'theta_offset': 0.0},
    {'alpha': -np.pi/2,  'a': 0.068, 'd': 0.000, 'theta_offset': -np.pi/2},
    {'alpha': 0.0,       'a': 0.111, 'd': 0.000, 'theta_offset': 0.0},
    {'alpha': 0.0,       'a': 0.137, 'd': 0.000, 'theta_offset': -np.pi/2},
    {'alpha': -np.pi/2,  'a': 0.000, 'd': 0.099, 'theta_offset': 0.0},
    {'alpha': 0.0,       'a': 0.000, 'd': 0.060, 'theta_offset': 0.0},
]

def dh_transform_matrix(theta, d, a, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct,     -st,     0.0,       a],
        [st*ca,  ct*ca,  -sa,  -sa*d],
        [st*sa,  ct*sa,   ca,   ca*d],
        [0.0,     0.0,   0.0,     1.0],
    ])

def rotation_matrix_to_rpy(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])

def calculate_forward_kinematics(joint_angles, dh_params):
    q = np.asarray(joint_angles, dtype=float)
    positions = [np.array([0.0, 0.0, 0.0, 1.0])]
    T = np.eye(4)

    for i, params in enumerate(dh_params):
        joint_theta = q[i] if i < len(q) else 0.0
        theta = joint_theta + params['theta_offset']
        T = T @ dh_transform_matrix(theta, params['d'], params['a'], params['alpha'])
        positions.append(T[:, 3])

    return np.array(positions)[:, :3], T

def rpy_to_rotation_matrix(roll, pitch, yaw):
    """
    Roll, Pitch, Yaw (Degree)를 X-Y-Z 고정축 회전 행렬로 변환합니다.
    """
    r, p, y_ang = np.radians([roll, pitch, yaw])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    Rz = np.array([
        [np.cos(y_ang), -np.sin(y_ang), 0],
        [np.sin(y_ang), np.cos(y_ang), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def get_pose(joint_angles):
    """
    관절 각도에 따른 End-Effector의 위치와 회전 행렬을 반환합니다.
    """
    positions, T = calculate_forward_kinematics(joint_angles, soarm_dh_params)
    pos = T[:3, 3]
    R = T[:3, :3]
    return pos, R

def compute_jacobian(joint_angles, delta=1e-5):
    """
    수치적 미분(Finite Difference)을 통해 6x5 자코비안 행렬을 계산합니다.
    SO-ARM101 Pro는 5-DOF이므로 J는 (6 x 5) 행렬입니다.
    """
    n_joints = len(joint_angles)
    J = np.zeros((6, n_joints))
    
    pos_curr, R_curr = get_pose(joint_angles)
    
    for i in range(n_joints):
        q_eps = joint_angles.copy()
        q_eps[i] += delta
        
        pos_eps, R_eps = get_pose(q_eps)
        
        # 1. 위치 자코비안 (선형 속도)
        J[:3, i] = (pos_eps - pos_curr) / delta
        
        # 2. 자세 자코비안 (각속도)
        # R_err = R_eps * R_curr^T
        R_err = R_eps @ R_curr.T
        
        # 회전 행렬을 Axis-Angle(회전축*각도) 형태의 오차 벡터로 변환
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
        if angle < 1e-6:
            w = np.zeros(3)
        else:
            w = (angle / (2 * np.sin(angle))) * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])
            
        J[3:, i] = w / delta
        
    return J

def numerical_ik_step(current_angles, target_pos, target_R, damping=0.05):
    """
    Damped Least Squares (DLS) 알고리즘을 사용한 IK 1 Step.
    5-DOF 로봇은 over-constrained 없이 위치(3) + 자세(3)을 모두 제어하면
    자유도가 부족할 수 있으므로, damping 값을 적절히 조정합니다.
    """
    pos_curr, R_curr = get_pose(current_angles)
    
    # 위치 오차
    err_pos = target_pos - pos_curr
    
    # 자세 오차 계산
    R_err = target_R @ R_curr.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        err_ori = np.zeros(3)
    else:
        err_ori = (angle / (2 * np.sin(angle))) * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        
    error_vector = np.concatenate([err_pos, err_ori])
    
    # 자코비안 계산
    J = compute_jacobian(current_angles)
    
    # DLS 식: delta_q = J^T * (J * J^T + lambda^2 * I)^-1 * error
    # 5-DOF에서는 역행렬 대신 pseudo-inverse(의사역행렬) 사용이 더 안정적
    J_T = J.T
    H = J_T @ J + (damping**2) * np.eye(len(current_angles))
    delta_q = np.linalg.solve(H, J_T @ error_vector)
    
    next_q = current_angles + delta_q
    # 관절 각도를 [-pi, pi] 범위로 정규화하여 각도가 무한히 커지는 현상 방지
    next_q = (next_q + np.pi) % (2 * np.pi) - np.pi
    
    return next_q, np.linalg.norm(error_vector)

if __name__ == "__main__":
    # 초기 로봇 포즈 (특이점을 피하기 위해 완전히 펴진 상태가 아닌 상태로 시작)
    init_q = np.radians([0.0, -60.0, 80.0, -20.0, 0.0])
    current_q = init_q.copy()
    
    # 초기 포즈의 FK 좌표를 타겟 초기값으로 설정
    init_pos, init_R = get_pose(init_q)
    init_rpy = rotation_matrix_to_rpy(init_R)

    # --- UI 구성 ---
    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title('Inverse Kinematics (SO-ARM101 Pro) - Numerical DLS')
    
    # 왼쪽: 3D Plot
    ax = fig.add_axes([0.05, 0.4, 0.45, 0.55], projection='3d')
    line, = ax.plot([], [], [], '-o', linewidth=3, markersize=8, label='SO-ARM101 Pro', color='royalblue')
    target_scatter = ax.scatter([], [], [], color='green', s=80, marker='*', label='Target Pose', zorder=6)
    ee_scatter = ax.scatter([], [], [], color='red', s=60, label='TCP (Actual)', zorder=5)
    title_ = ax.set_title('Inverse Kinematics (SO-ARM101 Pro)')
    
    ax.set_xlim([-0.45, 0.45])
    ax.set_ylim([-0.45, 0.45])
    ax.set_zlim([0.0, 0.65])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right')

    # 오른쪽: 텍스트 정보 창
    ax_text = fig.add_axes([0.55, 0.4, 0.42, 0.55])
    ax_text.axis('off')
    info_text = ax_text.text(0.0, 0.5, '', fontsize=10, verticalalignment='center', fontfamily='monospace')

    # --- 슬라이더 UI 구성 (하단) ---
    sliders = {}
    slider_labels = ['Target X (m)', 'Target Y (m)', 'Target Z (m)',
                     'Target Roll (deg)', 'Target Pitch (deg)', 'Target Yaw (deg)']
    slider_inits = [init_pos[0], init_pos[1], init_pos[2], init_rpy[0], init_rpy[1], init_rpy[2]]
    
    # SO-ARM101 Pro 작업 영역에 맞는 슬라이더 범위
    slider_bounds = [
        (-0.45, 0.45), (-0.45, 0.45), (0.0, 0.60),   # XYZ bounds
        (-180, 180), (-180, 180), (-180, 180)           # RPY bounds
    ]
    
    for i, label in enumerate(slider_labels):
        ax_s = fig.add_axes([0.15, 0.3 - i*0.045, 0.7, 0.03])
        s = Slider(ax_s, label, slider_bounds[i][0], slider_bounds[i][1], valinit=slider_inits[i])
        sliders[label] = s

    # --- 업데이트 콜백 ---
    is_updating = False

    def update(val):
        global current_q, is_updating
        if is_updating: return
        is_updating = True

        # 목표 좌표와 RPY 읽어오기
        target_p = np.array([sliders['Target X (m)'].val, sliders['Target Y (m)'].val, sliders['Target Z (m)'].val])
        target_rpy = [sliders['Target Roll (deg)'].val, sliders['Target Pitch (deg)'].val, sliders['Target Yaw (deg)'].val]
        target_Rot = rpy_to_rotation_matrix(*target_rpy)
        
        # IK 연산 (수치적 최적화 반복)
        max_iter = 100
        error_norm = 0
        for _ in range(max_iter):
            current_q, error_norm = numerical_ik_step(current_q, target_p, target_Rot, damping=0.05)
            # 오차가 충분히 작아지면 조기 종료
            if error_norm < 1e-4:
                break
                
        # 최종 계산된 Joint Angle로 다시 FK 실행하여 화면에 표시
        positions, T_final = calculate_forward_kinematics(current_q, soarm_dh_params)
        
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        target_scatter._offsets3d = ([target_p[0]], [target_p[1]], [target_p[2]])
        ee_scatter._offsets3d = ([x[-1]], [y[-1]], [z[-1]])
        
        # 텍스트 정보 업데이트
        q_deg = np.degrees(current_q)
        text_str = "=== Inverse Kinematics (SO-ARM101 Pro) ===\n\n"
        
        text_str += "[1. Target Pose (Sliders)]\n"
        text_str += f"X: {target_p[0]*1000:.1f}  Y: {target_p[1]*1000:.1f}  Z: {target_p[2]*1000:.1f} (mm)\n"
        text_str += f"R: {target_rpy[0]: 6.1f}  P: {target_rpy[1]: 6.1f}  Y: {target_rpy[2]: 6.1f} (deg)\n\n"
        
        text_str += "[2. Actual Robot Pose (FK Verification)]\n"
        curr_rpy = rotation_matrix_to_rpy(T_final[:3, :3])
        text_str += f"X: {x[-1]*1000:.1f}  Y: {y[-1]*1000:.1f}  Z: {z[-1]*1000:.1f} (mm)\n"
        text_str += f"R: {curr_rpy[0]: 6.1f}  P: {curr_rpy[1]: 6.1f}  Y: {curr_rpy[2]: 6.1f} (deg)\n\n"
        
        text_str += "[3. Calculated Joint Angles]\n"
        joint_names = ['J1 Shoulder Pan', 'J2 Shoulder Tilt', 'J3 Elbow',
                       'J4 Wrist Tilt', 'J5 Wrist Rotate']
        for i in range(5):
            text_str += f"{joint_names[i]}: {q_deg[i]: 8.2f} deg\n"
            
        text_str += f"\nError Norm: {error_norm:.5f}\n"
        if error_norm > 0.05:
            text_str += "!! WARNING: Target might be out of reach\n"
            text_str += "   or in singularity. Try a different pose."
            
        info_text.set_text(text_str)
        fig.canvas.draw_idle()
        
        is_updating = False

    for s in sliders.values():
        s.on_changed(update)

    # 초기 상태로 한 번 그리기
    update(0)

    plt.show()
