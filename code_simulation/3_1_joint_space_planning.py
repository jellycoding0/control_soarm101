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

def generate_joint_trajectory(q_start, q_end, steps=50):
    """
    Joint Space 상에서 두 관절 각도 사이를 선형 보간(Linear Interpolation)합니다.
    """
    trajectory = []
    for t in np.linspace(0, 1, steps):
        q_t = (1 - t) * q_start + t * q_end
        trajectory.append(q_t)
    return trajectory

if __name__ == "__main__":
    init_q = np.radians([0.0, -60.0, 80.0, -20.0, 0.0])
    current_q = init_q.copy()
    
    init_pos, init_R = get_pose(init_q)
    init_rpy = rotation_matrix_to_rpy(init_R)

    fig = plt.figure(figsize=(12, 9))
    fig.canvas.manager.set_window_title('Joint Space Planning (SO-ARM101 Pro)')
    
    ax = fig.add_axes([0.05, 0.4, 0.45, 0.5], projection='3d')
    line, = ax.plot([], [], [], '-o', linewidth=3, markersize=8, label='SO-ARM101 Pro', color='royalblue')
    target_scatter = ax.scatter([], [], [], color='green', s=40, marker='x', label='Target Pose', zorder=6)
    ee_scatter = ax.scatter([], [], [], color='red', s=15, label='End Effector', zorder=5)
    
    # 궤적을 그리기 위한 라인
    traj_line, = ax.plot([], [], [], '--', linewidth=1, color='orange', label='EE Trajectory')
    
    ax.set_title('Joint Space Planning (SO-ARM101 Pro)')
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
        (-0.45, 0.45), (-0.45, 0.45), (0.0, 0.60),
        (-180, 180), (-180, 180), (-180, 180)
    ]
    
    for i, label in enumerate(slider_labels):
        ax_s = fig.add_axes([0.15, 0.3 - i*0.045, 0.7, 0.03])
        s = Slider(ax_s, label, slider_bounds[i][0], slider_bounds[i][1], valinit=slider_inits[i])
        sliders[label] = s

    # Plan & Move 버튼 추가
    ax_button = fig.add_axes([0.4, 0.02, 0.2, 0.05])
    btn_plan = Button(ax_button, 'Plan & Move', hovercolor='0.975')

    def draw_robot(q):
        positions, T = calculate_forward_kinematics(q, soarm_dh_params)
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        ee_scatter._offsets3d = ([x[-1]], [y[-1]], [z[-1]])
        return positions[-1] # End-effector position

    def update_text(q, target_p, target_rpy, status_msg="Ready"):
        positions, T = calculate_forward_kinematics(q, soarm_dh_params)
        ee_pos = positions[-1]
        curr_rpy = rotation_matrix_to_rpy(T[:3, :3])
        q_deg = np.degrees(q)
        
        text_str = f"=== SO-ARM101 Pro Joint Planning ===\nStatus: {status_msg}\n\n"
        
        text_str += "[1. Target Pose (Goal)]\n"
        text_str += f"X: {target_p[0]:.3f}  Y: {target_p[1]:.3f}  Z: {target_p[2]:.3f} (m)\n"
        text_str += f"R: {target_rpy[0]: 6.1f}  P: {target_rpy[1]: 6.1f}  Y: {target_rpy[2]: 6.1f} (deg)\n\n"
        
        text_str += "[2. Actual Robot Pose]\n"
        text_str += f"X: {ee_pos[0]:.3f}  Y: {ee_pos[1]:.3f}  Z: {ee_pos[2]:.3f} (m)\n"
        text_str += f"R: {curr_rpy[0]: 6.1f}  P: {curr_rpy[1]: 6.1f}  Y: {curr_rpy[2]: 6.1f} (deg)\n\n"
        
        text_str += "[3. Current Joint Angles]\n"
        joint_names = ['J1 Base', 'J2 Shoulder', 'J3 Elbow', 'J4 Wrist Pitch', 'J5 Wrist Roll']
        for i in range(5):
            text_str += f"{joint_names[i]}: {q_deg[i]: 8.2f} deg\n"
            
        info_text.set_text(text_str)

    def on_slider_change(val):
        target_p = np.array([sliders['Target X (m)'].val, sliders['Target Y (m)'].val, sliders['Target Z (m)'].val])
        target_scatter._offsets3d = ([target_p[0]], [target_p[1]], [target_p[2]])
        
        target_rpy = [sliders['Target Roll (deg)'].val, sliders['Target Pitch (deg)'].val, sliders['Target Yaw (deg)'].val]
        update_text(current_q, target_p, target_rpy, status_msg="Target changed. Press 'Plan & Move'")
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

        # 1. 목표 포즈에 대한 IK 계산 (Target Joint Angles 도출)
        target_q, err = numerical_ik(current_q, target_p, target_Rot)

        if err > 0.05:
            update_text(current_q, target_p, target_rpy, status_msg=f"IK Error too high ({err:.3f}). Cannot plan.")
            is_moving = False
            return

        # 2. Joint Space Trajectory 생성 (선형 보간)
        steps = 40
        trajectory = generate_joint_trajectory(current_q, target_q, steps=steps)

        update_text(current_q, target_p, target_rpy, status_msg="Executing Joint Space Plan...")
        
        # 궤적 시각화를 위한 리스트
        ee_path_x, ee_path_y, ee_path_z = [], [], []

        # 3. 궤적에 따라 로봇 이동 애니메이션
        for i, q in enumerate(trajectory):
            ee_pos = draw_robot(q)
            
            ee_path_x.append(ee_pos[0])
            ee_path_y.append(ee_pos[1])
            ee_path_z.append(ee_pos[2])
            
            traj_line.set_data(ee_path_x, ee_path_y)
            traj_line.set_3d_properties(ee_path_z)
            
            update_text(q, target_p, target_rpy, status_msg=f"Moving... Step {i+1}/{steps}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.02) # 이동 속도 조절

        current_q = target_q
        update_text(current_q, target_p, target_rpy, status_msg="Movement Complete")
        is_moving = False

    btn_plan.on_clicked(on_plan_click)

    # 초기 상태 렌더링
    draw_robot(current_q)
    target_p = np.array([init_pos[0], init_pos[1], init_pos[2]])
    target_scatter._offsets3d = ([target_p[0]], [target_p[1]], [target_p[2]])
    update_text(current_q, target_p, init_rpy)

    plt.show()
