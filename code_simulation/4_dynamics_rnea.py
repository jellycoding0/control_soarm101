import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import time

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

def generate_cubic_trajectory(q_start, q_end, duration=2.0, steps=50):
    """
    3차 다항식(Cubic Polynomial)을 이용한 시작점-끝점 속도 0 기반 관절 궤적 생성
    """
    t_arr = np.linspace(0, duration, steps)
    q_traj, dq_traj, ddq_traj = [], [], []
    
    for t in t_arr:
        # 정규화된 시간 
        tau = t / duration
        
        # 3차 다항식 경계 조건 해 적용 (시작/끝 속도 0)
        s = 3 * tau**2 - 2 * tau**3
        ds = (6 * tau - 6 * tau**2) / duration
        dds = (6 - 12 * tau) / (duration**2)
        
        distance = q_end - q_start
        q_traj.append(q_start + distance * s)
        dq_traj.append(distance * ds)
        ddq_traj.append(distance * dds)
        
    return t_arr, q_traj, dq_traj, ddq_traj

def rpy_to_rotation_matrix(roll, pitch, yaw):
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
    positions, T = calculate_forward_kinematics(joint_angles, soarm_dh_params)
    pos = T[:3, 3]
    R = T[:3, :3]
    return pos, R

def compute_jacobian(joint_angles, delta=1e-5):
    n_joints = len(joint_angles)
    J = np.zeros((6, n_joints))
    pos_curr, R_curr = get_pose(joint_angles)
    
    for i in range(n_joints):
        q_eps = joint_angles.copy()
        q_eps[i] += delta
        pos_eps, R_eps = get_pose(q_eps)
        
        J[:3, i] = (pos_eps - pos_curr) / delta
        R_err = R_eps @ R_curr.T
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

def numerical_ik(current_angles, target_pos, target_R, max_iter=50, tol=1e-4, damping=0.01):
    q = current_angles.copy()
    for _ in range(max_iter):
        pos_curr, R_curr = get_pose(q)
        err_pos = target_pos - pos_curr
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
        if np.linalg.norm(error_vector) < tol:
            break
            
        J = compute_jacobian(q)
        J_T = J.T
        H = J_T @ J + (damping**2) * np.eye(len(q))
        delta_q = np.linalg.solve(H, J_T @ error_vector)
        q = q + delta_q
        q = (q + np.pi) % (2 * np.pi) - np.pi
    return q, np.linalg.norm(error_vector)

# --- 동역학 및 부드러운 궤적 생성 기능 추가 ---

def generate_trapezoidal_trajectory(q_start, q_end, duration=2.0, steps=50):
    """
    사다리꼴 속도 프로파일(Trapezoidal Velocity Profile)을 이용한 관절 궤적 생성
    가속, 등속, 감속 구간으로 나누어 부드러운 이동을 구현합니다.
    """
    t_arr = np.linspace(0, duration, steps)
    q_traj, dq_traj, ddq_traj = [], [], []
    
    # 가속/감속 시간을 전체 시간의 1/3로 설정
    t_a = duration / 3.0
    
    # 정규화된 최대 속도와 가속도 (이동 거리 1 기준)
    v_max = 1.0 / (duration - t_a)
    a_max = v_max / t_a
    
    distance = q_end - q_start
    
    for t in t_arr:
        if t <= t_a:
            s = 0.5 * a_max * t**2
            ds = a_max * t
            dds = a_max
        elif t <= duration - t_a:
            s = 0.5 * a_max * t_a**2 + v_max * (t - t_a)
            ds = v_max
            dds = 0.0
        else:
            t_dec = t - (duration - t_a)
            s = 0.5 * a_max * t_a**2 + v_max * (duration - 2*t_a) + v_max * t_dec - 0.5 * a_max * t_dec**2
            ds = v_max - a_max * t_dec
            dds = -a_max
            
        q_traj.append(q_start + distance * s)
        dq_traj.append(distance * ds)
        ddq_traj.append(distance * dds)
        
    return t_arr, q_traj, dq_traj, ddq_traj

# ============================================================
# SO-ARM101 Pro 동역학 파라미터 (질량, 무게중심, 관성모멘트)
# STS3215 서보 모터 기반 3D 프린팅 구조체 추정값
# (질량: 서보 포함 링크 전체 / COM: 링크 중앙 가정)
# ============================================================
soarm_dynamic_params = [
    {  # Joint 1: Shoulder Pan
        'mass': 0.180,
        'com': np.array([0.0, 0.0, 0.031]),       # Base 서보 높이 절반
        'inertia': np.diag([0.0002, 0.0002, 0.0001])
    },
    {  # Joint 2: Shoulder Tilt  (upper arm ~117mm)
        'mass': 0.210,
        'com': np.array([0.0585, 0.0, 0.0]),       # 링크 길이 절반
        'inertia': np.diag([0.00005, 0.00045, 0.00045])
    },
    {  # Joint 3: Elbow (forearm ~95.5mm)
        'mass': 0.165,
        'com': np.array([0.0478, 0.0, 0.0]),
        'inertia': np.diag([0.00003, 0.00025, 0.00025])
    },
    {  # Joint 4: Wrist Tilt
        'mass': 0.090,
        'com': np.array([0.0, 0.0, 0.0]),
        'inertia': np.diag([0.00001, 0.00001, 0.00001])
    },
    {  # Joint 5: Wrist Rotate + Gripper
        'mass': 0.120,
        'com': np.array([0.0, 0.0, 0.030]),        # TCP 방향 절반
        'inertia': np.diag([0.00002, 0.00002, 0.00001])
    },
]

def compute_inverse_dynamics_rnea(q, dq, ddq):
    """
    [동역학] Recursive Newton-Euler Algorithm (RNEA)
    SO-ARM101 Pro 5-DOF 버전
    입력: 관절 각도/속도/가속도 → 출력: 각 관절의 필요 토크
    """
    n = 5  # SO-ARM101 Pro는 5-DOF
    # 중력 보상을 위해 베이스를 Z축 양의 방향으로 가속시킵니다.
    g = np.array([0, 0, 9.81])
    
    # 0. DH Transform Matrices (soarm_dh_params 사용)
    A = []
    for i in range(n):
        d = soarm_dh_params[i]['d']
        a = soarm_dh_params[i]['a']
        alpha = soarm_dh_params[i]['alpha']
        theta = q[i] + soarm_dh_params[i]['theta_offset']
        A.append(dh_transform_matrix(theta, d, a, alpha))
        
    omega = []
    domega = []
    acc = []
    
    # Base conditions
    omega_prev = np.zeros(3)
    domega_prev = np.zeros(3)
    acc_prev = g.copy()
    z0 = np.array([0, 0, 1.0])
    
    # Forward Recursion (Base → End-Effector)
    for i in range(n):
        R_prev_i = A[i][:3, :3]
        R_i_prev = R_prev_i.T
        P_prev_i = A[i][:3, 3]
        
        omega_i = R_i_prev @ (omega_prev + dq[i] * z0)
        omega.append(omega_i)
        
        domega_i = R_i_prev @ (domega_prev + np.cross(omega_prev, dq[i] * z0) + ddq[i] * z0)
        domega.append(domega_i)
        
        acc_i = R_i_prev @ (acc_prev + np.cross(domega_prev, P_prev_i) + np.cross(omega_prev, np.cross(omega_prev, P_prev_i)))
        acc.append(acc_i)
        
        omega_prev = omega_i
        domega_prev = domega_i
        acc_prev = acc_i

    # Backward Recursion (End-Effector → Base)
    tau = np.zeros(n)
    f_next = np.zeros(3)
    n_next = np.zeros(3)
    
    for i in range(n - 1, -1, -1):
        mass = soarm_dynamic_params[i]['mass']
        com = soarm_dynamic_params[i]['com']
        inertia = soarm_dynamic_params[i]['inertia']
        
        omega_i = omega[i]
        domega_i = domega[i]
        acc_i = acc[i]
        
        acc_com_i = acc_i + np.cross(domega_i, com) + np.cross(omega_i, np.cross(omega_i, com))
        
        F_i = mass * acc_com_i
        N_i = inertia @ domega_i + np.cross(omega_i, inertia @ omega_i)
        
        R_prev_i = A[i][:3, :3]
        P_prev_i = A[i][:3, 3]
        p_i = R_prev_i.T @ P_prev_i
        
        if i < n - 1:
            R_i_next = A[i+1][:3, :3]
            f_next_i = R_i_next @ f_next
            n_next_i = R_i_next @ n_next
        else:
            f_next_i = np.zeros(3)
            n_next_i = np.zeros(3)
            
        f_i = f_next_i + F_i
        n_i = N_i + n_next_i + np.cross(p_i + com, F_i) + np.cross(p_i, f_next_i)
        
        n_prev = R_prev_i @ n_i
        
        # STS3215 서보 점성 마찰 계수 (추정값)
        viscous_friction = np.array([0.05, 0.05, 0.05, 0.02, 0.02])
        tau[i] = n_prev[2] + viscous_friction[i] * dq[i]
        
        f_next = f_i
        n_next = n_i
        
    return tau

def plot_dynamics_profiles(t_arr, q_traj, dq_traj, ddq_traj, tau_traj):
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    fig.canvas.manager.set_window_title('Dynamics Profiles (Position, Velocity, Accel, Torque)')
    
    q_traj = np.array(q_traj)
    dq_traj = np.array(dq_traj)
    ddq_traj = np.array(ddq_traj)
    tau_traj = np.array(tau_traj)
    
    labels = [f'J{i+1}' for i in range(5)]  # SO-ARM100은 5-DOF
    
    axs[0].plot(t_arr, np.degrees(q_traj))
    axs[0].set_ylabel('Position (deg)')
    axs[0].set_title('Joint Profiles during movement')
    axs[0].legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    axs[0].grid(True)
    
    axs[1].plot(t_arr, np.degrees(dq_traj))
    axs[1].set_ylabel('Velocity (deg/s)')
    axs[1].grid(True)
    
    axs[2].plot(t_arr, np.degrees(ddq_traj))
    axs[2].set_ylabel('Acceleration (deg/s^2)')
    axs[2].grid(True)
    
    axs[3].plot(t_arr, tau_traj)
    axs[3].set_ylabel('Torque (Nm)')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show(block=False)

if __name__ == "__main__":
    init_q = np.radians([0.0, -60.0, 80.0, -20.0, 0.0])
    current_q = init_q.copy()
    
    init_pos, init_R = get_pose(init_q)
    init_rpy = rotation_matrix_to_rpy(init_R)

    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title('Joint Trajectory Planning with Dynamics (SO-ARM101 Pro)')
    
    ax = fig.add_axes([0.05, 0.4, 0.45, 0.55], projection='3d')
    line, = ax.plot([], [], [], '-o', linewidth=3, markersize=8, label='SO-ARM101 Pro', color='royalblue')
    target_scatter = ax.scatter([], [], [], color='green', s=80, marker='*', label='Target Pose', zorder=6)
    ee_scatter = ax.scatter([], [], [], color='red', s=60, label='TCP (Actual)', zorder=5)
    
    traj_line, = ax.plot([], [], [], '--', linewidth=1.5, color='orange', label='TCP Trajectory')
    
    ax.set_title('Joint Planning & Dynamics (SO-ARM101 Pro)')
    ax.set_xlim([-0.45, 0.45])
    ax.set_ylim([-0.45, 0.45])
    ax.set_zlim([0.0, 0.65])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right')

    ax_text = fig.add_axes([0.55, 0.4, 0.4, 0.5])
    ax_text.axis('off')
    info_text = ax_text.text(0.0, 0.5, '', fontsize=11, verticalalignment='center', fontfamily='monospace')

    sliders = {}
    slider_labels = ['Target X (m)', 'Target Y (m)', 'Target Z (m)', 'Target Roll (deg)', 'Target Pitch (deg)', 'Target Yaw (deg)']
    slider_inits = [init_pos[0], init_pos[1], init_pos[2], init_rpy[0], init_rpy[1], init_rpy[2]]
    slider_bounds = [
        (-0.45, 0.45), (-0.45, 0.45), (0.0, 0.60),  # SO-ARM101 Pro 작업 영역
        (-180, 180), (-180, 180), (-180, 180)
    ]
    
    for i, label in enumerate(slider_labels):
        ax_s = fig.add_axes([0.15, 0.3 - i*0.045, 0.7, 0.03])
        s = Slider(ax_s, label, slider_bounds[i][0], slider_bounds[i][1], valinit=slider_inits[i])
        sliders[label] = s

    ax_button = fig.add_axes([0.4, 0.02, 0.2, 0.05])
    btn_plan = Button(ax_button, 'Plan, Move & Calc Dynamics', hovercolor='0.975')

    def draw_robot(q):
        positions, T = calculate_forward_kinematics(q, soarm_dh_params)
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        ee_scatter._offsets3d = ([x[-1]], [y[-1]], [z[-1]])
        return positions[-1]

    def update_text(q, target_p, target_rpy, status_msg="Ready", torque_msg=""):
        positions, T = calculate_forward_kinematics(q, soarm_dh_params)
        ee_pos = positions[-1]
        curr_rpy = rotation_matrix_to_rpy(T[:3, :3])
        q_deg = np.degrees(q)
        
        text_str = f"=== SO-ARM101 Pro Planning & Dynamics ===\nStatus: {status_msg}\n\n"
        
        text_str += "[1. Actual Robot Pose]\n"
        text_str += f"X: {ee_pos[0]*1000:.1f}  Y: {ee_pos[1]*1000:.1f}  Z: {ee_pos[2]*1000:.1f} (mm)\n"
        text_str += f"R: {curr_rpy[0]: 6.1f}  P: {curr_rpy[1]: 6.1f}  Y: {curr_rpy[2]: 6.1f} (deg)\n\n"
        
        text_str += "[2. Current Joint Angles]\n"
        joint_names = ['J1 Pan', 'J2 Tilt', 'J3 Elbow', 'J4 Wrist', 'J5 Rot']
        for i in range(5):
            text_str += f"{joint_names[i]}: {q_deg[i]: 7.2f} deg  "
            if i % 2 == 1: text_str += "\n"
        text_str += "\n"
            
        if not torque_msg:
            _tau = compute_inverse_dynamics_rnea(q, np.zeros(5), np.zeros(5))
            torque_msg = "Required Torques (Nm):\n"
            for j in range(5):
                torque_msg += f"T{j+1}: {_tau[j]: 5.2f} "
                if j % 2 == 1: torque_msg += "\n"
            
        if torque_msg:
            text_str += f"\n[3. Dynamics Info]\n{torque_msg}"
            
        info_text.set_text(text_str)

    def on_slider_change(val):
        target_p = np.array([sliders['Target X (m)'].val, sliders['Target Y (m)'].val, sliders['Target Z (m)'].val])
        target_scatter._offsets3d = ([target_p[0]], [target_p[1]], [target_p[2]])
        
        target_rpy = [sliders['Target Roll (deg)'].val, sliders['Target Pitch (deg)'].val, sliders['Target Yaw (deg)'].val]
        update_text(current_q, target_p, target_rpy, status_msg="Target changed. Press 'Plan, Move...'")
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(on_slider_change)

    is_moving = False

    def on_plan_click(event):
        global current_q, is_moving
        if is_moving: return
        is_moving = True

        target_p = np.array([sliders['Target X (m)'].val, sliders['Target Y (m)'].val, sliders['Target Z (m)'].val])
        target_rpy = [sliders['Target Roll (deg)'].val, sliders['Target Pitch (deg)'].val, sliders['Target Yaw (deg)'].val]
        target_Rot = rpy_to_rotation_matrix(*target_rpy)

        update_text(current_q, target_p, target_rpy, status_msg="Calculating IK...")
        fig.canvas.draw()

        target_q, err = numerical_ik(current_q, target_p, target_Rot)

        if err > 0.05:
            update_text(current_q, target_p, target_rpy, status_msg=f"IK Error ({err:.3f}). Cannot plan.")
            is_moving = False
            return
        
        update_text(current_q, target_p, target_rpy, status_msg="Generating Trapezoidal Trajectory...")
        
        # [수정 1] 함수 이름을 상단에 정의한 generate_trapezoidal_trajectory 로 변경합니다.
        duration = 2.0
        steps = 40
        t_arr, q_traj, dq_traj, ddq_traj = generate_trapezoidal_trajectory(current_q, target_q, duration, steps)

        ee_path_x, ee_path_y, ee_path_z = [], [], []
        tau_traj = [] # [수정 2] 끊겨있던 빈 리스트 선언 코드를 완포(완성)합니다.

        update_text(current_q, target_p, target_rpy, status_msg="Moving and Calculating Dynamics...")
        
        # 2. 이동 및 토크 계산 애니메이션
        for i in range(steps):
            q, dq, ddq = q_traj[i], dq_traj[i], ddq_traj[i]
            
            # 역동역학을 통한 필요 토크(tau) 계산
            tau = compute_inverse_dynamics_rnea(q, dq, ddq)
            tau_traj.append(tau)
            
            ee_pos = draw_robot(q)
            ee_path_x.append(ee_pos[0])
            ee_path_y.append(ee_pos[1])
            ee_path_z.append(ee_pos[2])
            
            traj_line.set_data(ee_path_x, ee_path_y)
            traj_line.set_3d_properties(ee_path_z)
            
            torque_str = "Required Torques (Nm):\n"
            for j in range(5):
                torque_str += f"T{j+1}: {tau[j]: 5.2f} "
                if j % 2 == 1: torque_str += "\n"
                
            update_text(q, target_p, target_rpy, status_msg=f"Moving... {i+1}/{steps}", torque_msg=torque_str)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(duration / steps) 

        current_q = target_q
        update_text(current_q, target_p, target_rpy, status_msg="Movement Complete! Check Dynamics Plot.")
        
        # 3. 궤적이 끝난 후 동역학 프로파일(위치, 속도, 가속도, 토크) 플롯 띄우기
        plot_dynamics_profiles(t_arr, q_traj, dq_traj, ddq_traj, tau_traj)
        
        is_moving = False

    btn_plan.on_clicked(on_plan_click)

    draw_robot(current_q)
    target_p = np.array([init_pos[0], init_pos[1], init_pos[2]])
    target_scatter._offsets3d = ([target_p[0]], [target_p[1]], [target_p[2]])
    
    init_dq = np.zeros(5)
    init_ddq = np.zeros(5)
    init_tau = compute_inverse_dynamics_rnea(current_q, init_dq, init_ddq)
    
    torque_str = "Required Torques (Nm):\n"
    for j in range(5):
        torque_str += f"T{j+1}: {init_tau[j]: 5.2f} "
        if j % 2 == 1: torque_str += "\n"
        
    update_text(current_q, target_p, init_rpy, status_msg="Ready", torque_msg=torque_str)

    plt.show()
