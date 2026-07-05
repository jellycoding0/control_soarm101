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

def generate_linear_task_trajectory(start_pos, start_rpy, end_pos, end_rpy, start_q, steps=40):
    """
    Task Space 상에서 시작점과 끝점 사이를 직선(선형 보간)으로 이동하는 Joint 궤적 생성
    """
    trajectory_q = []
    current_q = start_q.copy()
    
    for t in np.linspace(0, 1, steps):
        pos_t = (1 - t) * start_pos + t * end_pos
        rpy_t = (1 - t) * np.array(start_rpy) + t * np.array(end_rpy)
        R_t = rpy_to_rotation_matrix(*rpy_t)
        
        # 이전 step의 q를 초기값으로 사용하여 IK 계산 (연속성 확보)
        q_t, err = numerical_ik(current_q, pos_t, R_t, max_iter=50)
        trajectory_q.append(q_t)
        current_q = q_t
        
    return trajectory_q

def generate_circular_task_trajectory(start_pos, start_rpy, end_pos, end_rpy, start_q, steps=40):
    """
    Task Space 상에서 시작점과 끝점을 지나는 위쪽 방향의 반원호(원호 보간) 궤적 생성
    """
    C = (start_pos + end_pos) / 2.0
    v1 = start_pos - C
    r = np.linalg.norm(v1)
    
    if r < 1e-4: # 시작점과 끝점이 같으면 직선 플래닝으로 대체
        return generate_linear_task_trajectory(start_pos, start_rpy, end_pos, end_rpy, start_q, steps)
        
    # v1과 수직이면서 가급적 Z축(위쪽)을 향하는 벡터 v2 탐색
    v_dir = v1 / r
    z_vec = np.array([0.0, 0.0, 1.0])
    v_proj = z_vec - np.dot(z_vec, v_dir) * v_dir
    
    if np.linalg.norm(v_proj) < 1e-4:
        # v1이 이미 Z축과 평행하다면 Y축 방향을 사용
        y_vec = np.array([0.0, 1.0, 0.0])
        v_proj = y_vec - np.dot(y_vec, v_dir) * v_dir
        
    v2 = (v_proj / np.linalg.norm(v_proj)) * r
    
    trajectory_q = []
    current_q = start_q.copy()
    
    for t in np.linspace(0, 1, steps):
        theta = t * np.pi # 0에서 pi까지 반원
        pos_t = C + np.cos(theta) * v1 + np.sin(theta) * v2
        rpy_t = (1 - t) * np.array(start_rpy) + t * np.array(end_rpy)
        R_t = rpy_to_rotation_matrix(*rpy_t)
        
        q_t, err = numerical_ik(current_q, pos_t, R_t, max_iter=50)
        trajectory_q.append(q_t)
        current_q = q_t
        
    return trajectory_q

if __name__ == "__main__":
    init_q = np.radians([0.0, -60.0, 80.0, -20.0, 0.0])
    current_q = init_q.copy()
    
    init_pos, init_R = get_pose(init_q)
    init_rpy = rotation_matrix_to_rpy(init_R)

    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title('Task Space Cartesian Planning (SO-ARM101 Pro)')
    
    ax = fig.add_axes([0.05, 0.4, 0.45, 0.55], projection='3d')
    line, = ax.plot([], [], [], '-o', linewidth=3, markersize=8, label='SO-ARM101 Pro', color='royalblue')
    target_scatter = ax.scatter([], [], [], color='green', s=80, marker='*', label='Target Pose', zorder=6)
    ee_scatter = ax.scatter([], [], [], color='red', s=60, label='TCP (Actual)', zorder=5)
    
    traj_line, = ax.plot([], [], [], '--', linewidth=1.5, color='orange', label='TCP Trajectory')
    
    ax.set_title('Task Space Planning - SO-ARM101 Pro\n(Linear & Circular Interpolation)')
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

    # 2개의 버튼: Line (직선), Circle (원호)
    ax_btn_line = fig.add_axes([0.3, 0.02, 0.15, 0.05])
    btn_line = Button(ax_btn_line, 'Move Line', hovercolor='0.975')

    ax_btn_circle = fig.add_axes([0.5, 0.02, 0.15, 0.05])
    btn_circle = Button(ax_btn_circle, 'Move Circle', hovercolor='0.975')

    def draw_robot(q):
        positions, T = calculate_forward_kinematics(q, soarm_dh_params)
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        ee_scatter._offsets3d = ([x[-1]], [y[-1]], [z[-1]])
        return positions[-1]

    def update_text(q, target_p, target_rpy, status_msg="Ready", plan_type="None"):
        positions, T = calculate_forward_kinematics(q, soarm_dh_params)
        ee_pos = positions[-1]
        curr_rpy = rotation_matrix_to_rpy(T[:3, :3])
        q_deg = np.degrees(q)
        
        text_str = f"=== Task Space Planning (SO-ARM101 Pro) ===\nMode: {plan_type}\nStatus: {status_msg}\n\n"
        
        text_str += "[1. Target Pose (Goal)]\n"
        text_str += f"X: {target_p[0]*1000:.1f}  Y: {target_p[1]*1000:.1f}  Z: {target_p[2]*1000:.1f} (mm)\n"
        text_str += f"R: {target_rpy[0]: 6.1f}  P: {target_rpy[1]: 6.1f}  Y: {target_rpy[2]: 6.1f} (deg)\n\n"
        
        text_str += "[2. Actual Robot Pose]\n"
        text_str += f"X: {ee_pos[0]*1000:.1f}  Y: {ee_pos[1]*1000:.1f}  Z: {ee_pos[2]*1000:.1f} (mm)\n"
        text_str += f"R: {curr_rpy[0]: 6.1f}  P: {curr_rpy[1]: 6.1f}  Y: {curr_rpy[2]: 6.1f} (deg)\n\n"
        
        text_str += "[3. Current Joint Angles]\n"
        joint_names = ['J1 Shoulder Pan', 'J2 Shoulder Tilt', 'J3 Elbow',
                       'J4 Wrist Tilt', 'J5 Wrist Rotate']
        for i in range(5):
            text_str += f"{joint_names[i]}: {q_deg[i]: 8.2f} deg\n"
            
        info_text.set_text(text_str)

    def on_slider_change(val):
        target_p = np.array([sliders['Target X (m)'].val, sliders['Target Y (m)'].val, sliders['Target Z (m)'].val])
        target_scatter._offsets3d = ([target_p[0]], [target_p[1]], [target_p[2]])
        target_rpy = [sliders['Target Roll (deg)'].val, sliders['Target Pitch (deg)'].val, sliders['Target Yaw (deg)'].val]
        update_text(current_q, target_p, target_rpy, status_msg="Target changed. Select planning mode.")
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(on_slider_change)

    is_moving = False

    def execute_trajectory(plan_type):
        global current_q, is_moving
        if is_moving: return
        is_moving = True

        target_p = np.array([sliders['Target X (m)'].val, sliders['Target Y (m)'].val, sliders['Target Z (m)'].val])
        target_rpy = [sliders['Target Roll (deg)'].val, sliders['Target Pitch (deg)'].val, sliders['Target Yaw (deg)'].val]
        target_Rot = rpy_to_rotation_matrix(*target_rpy)

        # 현재 자세 구하기
        start_pos, start_R = get_pose(current_q)
        start_rpy = rotation_matrix_to_rpy(start_R)

        update_text(current_q, target_p, target_rpy, status_msg="Generating trajectory...", plan_type=plan_type)
        fig.canvas.draw()

        # 목표가 도달 가능한지 최종 IK로 먼저 검증
        final_q, err = numerical_ik(current_q, target_p, target_Rot)
        if err > 0.05:
            update_text(current_q, target_p, target_rpy, status_msg=f"Target unreachable (Error: {err:.3f}).", plan_type=plan_type)
            is_moving = False
            return

        steps = 40
        if plan_type == "Line":
            trajectory = generate_linear_task_trajectory(start_pos, start_rpy, target_p, target_rpy, current_q, steps)
        else: # Circle
            trajectory = generate_circular_task_trajectory(start_pos, start_rpy, target_p, target_rpy, current_q, steps)

        update_text(current_q, target_p, target_rpy, status_msg="Executing trajectory...", plan_type=plan_type)
        
        ee_path_x, ee_path_y, ee_path_z = [], [], []

        for i, q in enumerate(trajectory):
            ee_pos = draw_robot(q)
            
            ee_path_x.append(ee_pos[0])
            ee_path_y.append(ee_pos[1])
            ee_path_z.append(ee_pos[2])
            
            traj_line.set_data(ee_path_x, ee_path_y)
            traj_line.set_3d_properties(ee_path_z)
            
            update_text(q, target_p, target_rpy, status_msg=f"Moving... Step {i+1}/{steps}", plan_type=plan_type)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.02) 

        current_q = trajectory[-1]
        update_text(current_q, target_p, target_rpy, status_msg="Movement Complete", plan_type=plan_type)
        is_moving = False

    def on_line_click(event):
        execute_trajectory("Line")

    def on_circle_click(event):
        execute_trajectory("Circle")

    btn_line.on_clicked(on_line_click)
    btn_circle.on_clicked(on_circle_click)

    # 초기 상태 렌더링
    draw_robot(current_q)
    target_p = np.array([init_pos[0], init_pos[1], init_pos[2]])
    target_scatter._offsets3d = ([target_p[0]], [target_p[1]], [target_p[2]])
    update_text(current_q, target_p, init_rpy)

    plt.show()
