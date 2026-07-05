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

if __name__ == "__main__":
    # 초기 관절 각도 (Degree)
    init_degrees = [0.0, -60.0, 80.0, -20.0, 0.0]

    # --- UI 구성 ---
    fig = plt.figure(figsize=(10, 9))
    fig.canvas.manager.set_window_title('Forward Kinematics (SO-ARM101 Pro)')
    
    # 3D Plot 영역 설정
    ax = fig.add_axes([0.05, 0.4, 0.9, 0.55], projection='3d')
    
    # 로봇 시각화 객체 초기화 (데이터는 update 함수에서 갱신)
    line, = ax.plot([], [], [], '-o', linewidth=3, markersize=8, label='SO-ARM101 Pro', color='royalblue')
    base_scatter = ax.scatter([], [], [], color='black', s=50, label='Base', zorder=5)
    ee_scatter = ax.scatter([], [], [], color='red', s=80, label='End Effector (TCP)', zorder=5)
    title_ = ax.set_title('Forward Kinematics (SO-ARM101 Pro)\n')
    
    # 그래프 축 범위 및 라벨 고정
    ax.set_xlim([-0.45, 0.45])
    ax.set_ylim([-0.45, 0.45])
    ax.set_zlim([0.0, 0.65])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right')

    # --- 슬라이더 UI 구성 ---
    sliders = []
    joint_limits = [
        (-180, 180),  # J1: Shoulder Pan
        (-120, 120),  # J2: Shoulder Tilt
        (-120, 120),  # J3: Elbow
        (-120, 120),  # J4: Wrist Tilt
        (-180, 180),  # J5: Wrist Rotate
    ]
    joint_names = ['J1 Shoulder Pan (deg)', 'J2 Shoulder Tilt (deg)', 'J3 Elbow (deg)',
                   'J4 Wrist Tilt (deg)', 'J5 Wrist Rotate (deg)']

    for i in range(5):
        ax_s = fig.add_axes([0.2, 0.30 - i*0.055, 0.65, 0.03])
        slider = Slider(
            ax=ax_s,
            label=joint_names[i],
            valmin=joint_limits[i][0],
            valmax=joint_limits[i][1],
            valinit=init_degrees[i],
        )
        sliders.append(slider)

    # --- 이벤트 업데이트 콜백 ---
    def update(val):
        # 1. 슬라이더에서 현재 관절 각도(rad) 읽기
        current_degrees = [s.val for s in sliders]
        current_angles_rad = np.radians(current_degrees)
        
        # 2. 정기구학 계산을 통해 각 링크 위치 및 최종 변환 행렬 획득
        positions, T = calculate_forward_kinematics(current_angles_rad, soarm_dh_params)
        
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        # 3. 3D 로봇 팔 데이터 갱신
        line.set_data(x, y)
        line.set_3d_properties(z)
        
        # 4. Base 및 End-Effector 마커 갱신
        base_scatter._offsets3d = ([x[0]], [y[0]], [z[0]])
        ee_scatter._offsets3d = ([x[-1]], [y[-1]], [z[-1]])
        
        # 5. End-Effector의 RPY 각도 계산
        rotation_matrix = T[:3, :3]
        roll, pitch, yaw = rotation_matrix_to_rpy(rotation_matrix)
        
        # 6. 상단 제목에 실시간 위치 및 자세(RPY) 텍스트 갱신
        title_.set_text(
            f'Forward Kinematics (SO-ARM101 Pro)\n'
            f'TCP Position [mm]: X={x[-1]*1000:.1f},  Y={y[-1]*1000:.1f},  Z={z[-1]*1000:.1f}\n'
            f'TCP Orientation [deg]: R={roll:.1f},  P={pitch:.1f},  Y={yaw:.1f}'
        )
        # 화면 재표시
        fig.canvas.draw_idle()

    # 슬라이더 이벤트 연결
    for s in sliders:
        s.on_changed(update)

    # 초기 화면 렌더링
    update(0)
    plt.show()


