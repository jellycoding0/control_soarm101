import numpy as np
import matplotlib.pyplot as plt
import time

# Import modularized components to maintain backwards compatibility
from so_servo_bus import (
    FeetechServoBus,
    DEFAULT_JOINT_OFFSETS_DEG,
    DEFAULT_PORT,
    DEFAULT_BAUDRATE,
    DEFAULT_JOINT_DIRECTIONS,
    MAX_FIRST_MOVE_STEP,
    JOINT_LIMITS_DEG,
    CONFIG_DIR,
    ROBOT_CONFIG_FILE,
    RESTING_CALIBRATION_FILE,
    HOME_VERTICAL_CALIBRATION_FILE,
    normalize_deg,
    load_robot_config,
    ROBOT_CONFIG,
    CONFIG_PORT,
    CONFIG_BAUDRATE,
    JOINT_DIRECTIONS,
    load_joint_offsets_deg,
    load_kinematic_zero_offsets_deg
)
from so_kinematics import Kinematics
from so_trajectory import TrajectoryGenerator


class SoArm101ProController:
    def __init__(self, port=None, simulate=False, dt=0.02, joint_offsets_deg=None):
        self.dt = dt
        self.simulate = simulate
        self.joint_ids = [1, 2, 3, 4, 5]
        self.all_ids = [1, 2, 3, 4, 5, 6]  # 그리퍼 포함 전체 아이디
        self.current_joints = np.zeros(5)
        self.motion_ready = False
        self.current_gripper_deg = 0.0
        
        self.bus = FeetechServoBus(port, simulate=simulate, joint_offsets_deg=joint_offsets_deg)
        self.kinematics = Kinematics()
        self.traj_gen = TrajectoryGenerator()
        
        if self.simulate:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        self.bus.open()
        if not self.bus.simulate:
            self.sync_current_joints_from_servos()
            print("안전 모드: open()에서는 토크를 자동으로 켜지 않습니다.")

    def close(self):
        #  종료할 때 안전을 위해 모터 토크를 풀어서 흐물흐물하게 만듭니다.
        if not self.bus.simulate:
            self.bus.set_torque(self.all_ids, enable=False)
        self.motion_ready = False
        self.bus.close()
        if self.simulate:
            plt.ioff()
            plt.show()

    def _update_and_drive(self, q):
        self.current_joints = np.array(q)
        self.bus.sync_write_positions(self.joint_ids, self.current_joints)
        
        if self.simulate:
            self._render_robot()
            time.sleep(self.dt)
        else:
            # 실제 구동 시에는 무조건 dt 간격을 유지하며 하드웨어 주기 동기화
            time.sleep(self.dt)

    def _present_positions(self):
        return [self.bus.read_present_position(j_id) for j_id in self.joint_ids]

    def _target_positions(self, q):
        return [
            self.bus.rad_to_pos(rad, j_id)
            for j_id, rad in zip(self.joint_ids, q)
        ]

    def _prepare_motion_safely(self, target_joints):
        if self.bus.simulate:
            return
        if self.motion_ready:
            return

        present_positions = self._present_positions()
        current_q = np.array([
            self.bus.pos_to_rad(pos, j_id)
            for j_id, pos in zip(self.joint_ids, present_positions)
        ])
        target_positions = self._target_positions(target_joints)
        deltas = [
            target - present
            for target, present in zip(target_positions, present_positions)
        ]

        print("이동 전 position 검사:")
        for j_id, present, target, delta in zip(self.joint_ids, present_positions, target_positions, deltas):
            print(f"  J{j_id}: present={present}, target={target}, delta={delta}")

        too_large = [
            (j_id, delta)
            for j_id, delta in zip(self.joint_ids, deltas)
            if abs(delta) > MAX_FIRST_MOVE_STEP
        ]
        if too_large:
            raise RuntimeError(
                "위험한 큰 이동 감지. 모터 방향/오프셋을 먼저 확인하세요: "
                + ", ".join(f"J{j_id} delta={delta}" for j_id, delta in too_large)
            )

        self.current_joints = current_q
        for j_id, rad in zip(self.joint_ids, self.current_joints):
            self.bus.sync_write_positions([j_id], [rad])
            time.sleep(0.01)
            self.bus.set_torque([j_id], enable=True)
            self.bus.sync_write_positions([j_id], [rad])
            time.sleep(0.02)
        self.motion_ready = True

    def sync_current_joints_from_servos(self):
        positions = [self.bus.read_present_position(j_id) for j_id in self.joint_ids]
        self.current_joints = np.array([
            self.bus.pos_to_rad(pos, j_id)
            for j_id, pos in zip(self.joint_ids, positions)
        ])
        degs = [round(float(np.rad2deg(rad)), 3) for rad in self.current_joints]
        print(f"현재 관절각 동기화 완료(deg): {degs}")
        return self.current_joints

    def _check_joint_limits(self, q):
        q_deg = np.rad2deg(q)
        for idx, (deg, (low, high)) in enumerate(zip(q_deg, JOINT_LIMITS_DEG), start=1):
            if deg < low or deg > high:
                raise ValueError(
                    f"J{idx} 관절각 {deg:.2f}°가 소프트 리밋 [{low:.1f}, {high:.1f}]°를 벗어났습니다."
                )

    def _render_robot(self):
        self.ax.cla()
        _, frames = self.kinematics.forward_kinematics(self.current_joints)
        
        # 1. Floor grid (Z = 0) representing the work table
        x_grid, y_grid = np.meshgrid(np.linspace(-300, 300, 7), np.linspace(-300, 300, 7))
        z_grid = np.zeros_like(x_grid)
        self.ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightgray', alpha=0.7, linewidth=0.8)
        
        # 2. Base pedestal (Z = 0 to Z = 119)
        shoulder_height = frames[1][2, 3]
        self.ax.plot([0, 0], [0, 0], [0, shoulder_height], color='dimgray', linewidth=10, solid_capstyle='round', label='Base Pedestal')
        
        # Colors for each joint link segment
        link_colors = [
            'darkorange',   # Segment 1 (J1 -> J2)
            'limegreen',    # Segment 2 (J2 -> J3)
            'royalblue',    # Segment 3 (J3 -> J4)
            'mediumpurple', # Segment 4 (J4 -> J5)
            'teal'          # Segment 5 (J5 -> J6)
        ]
        
        # 3. Draw each link segment with a unique color
        for i in range(1, 6):
            p_start = frames[i]
            p_end = frames[i + 1]
            self.ax.plot(
                [p_start[0, 3], p_end[0, 3]],
                [p_start[1, 3], p_end[1, 3]],
                [p_start[2, 3], p_end[2, 3]],
                '-o',
                color=link_colors[i - 1],
                linewidth=5,
                markersize=7,
                solid_capstyle='round'
            )
            
        # Highlight End-Effector
        self.ax.scatter(frames[-1][0, 3], frames[-1][1, 3], frames[-1][2, 3], color='crimson', s=100, label='TCP', zorder=10)
        
        u_x = frames[6][:3, 1]  # Sideways direction of the fingers (Y-axis for 90 deg rotation)
        u_z = frames[6][:3, 2]  # Forward direction of the fingers
        p_base = frames[6][:3, 3]  # Base of the gripper
        
        # Map current_gripper_deg to opening width (5mm to 50mm)
        g_min, g_max = -60.0, 60.0
        w_open = 5.0 + 45.0 * (self.current_gripper_deg - g_min) / (g_max - g_min)
        w_base = 25.0
        l_finger = 35.0
        
        p_left_base = p_base - (w_base / 2.0) * u_x
        p_right_base = p_base + (w_base / 2.0) * u_x
        p_left_tip = p_base + l_finger * u_z - (w_open / 2.0) * u_x
        p_right_tip = p_base + l_finger * u_z + (w_open / 2.0) * u_x
        
        # Draw gripper yoke
        self.ax.plot(
            [p_left_base[0], p_right_base[0]],
            [p_left_base[1], p_right_base[1]],
            [p_left_base[2], p_right_base[2]],
            color='darkslategray',
            linewidth=3
        )
        # Draw left finger
        self.ax.plot(
            [p_left_base[0], p_left_tip[0]],
            [p_left_base[1], p_left_tip[1]],
            [p_left_base[2], p_left_tip[2]],
            color='darkslategray',
            linewidth=3,
            solid_capstyle='round'
        )
        # Draw right finger
        self.ax.plot(
            [p_right_base[0], p_right_tip[0]],
            [p_right_base[1], p_right_tip[1]],
            [p_right_base[2], p_right_tip[2]],
            color='darkslategray',
            linewidth=3,
            solid_capstyle='round'
        )
        
        # 5. Add text labels to each joint axis index
        joint_names = ["J1", "J2", "J3", "J4", "J5", "TCP"]
        for i in range(1, 7):
            p = frames[i][:3, 3]
            self.ax.text(
                p[0] + 8,
                p[1] + 8,
                p[2] + 8,
                joint_names[i - 1],
                color='black',
                fontsize=9,
                fontweight='bold'
            )
            
        self.ax.set_xlim([-300, 300])
        self.ax.set_ylim([-300, 300])
        self.ax.set_zlim([-20, 500])
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.fig.canvas.flush_events()

    def move_joints(self, target_joints, duration=1.5):
        target_joints = np.array(target_joints, dtype=float)
        self._check_joint_limits(target_joints)
        self._prepare_motion_safely(target_joints)
        trajectory = self.traj_gen.joint_trajectory(self.current_joints, target_joints, duration, self.dt)
        for q in trajectory:
            self._update_and_drive(q)

    def move_to_xyz(self, xyz, duration=2.0):
        target_q = self.kinematics.inverse_kinematics_position_dls(xyz, self.current_joints)
        self.move_joints(target_q, duration)

    def home(self):
        print("[HOME] 홈 포지션으로 이동합니다.")
        self.move_joints([0.0, 0.0, 0.0, 0.0, 0.0], duration=2.0)

    def open_gripper(self):
        self.current_gripper_deg = 45.0
        if not self.bus.simulate:
            self.bus.set_torque([6], enable=True)
            self.bus.sync_write_positions([6], [np.deg2rad(45)])
        print("[OPEN] 그리퍼 개방 (ID 6)")
        if self.simulate:
            self._render_robot()

    def close_gripper(self):
        self.current_gripper_deg = -45.0
        if not self.bus.simulate:
            self.bus.set_torque([6], enable=True)
            self.bus.sync_write_positions([6], [np.deg2rad(-45)])
        print("[CLOSE] 그리퍼 폐쇄 (ID 6)")
        if self.simulate:
            self._render_robot()


if __name__ == "__main__":
    with SoArm101ProController(simulate=False, dt=0.02) as arm:
        print("\n현재 자세 동기화만 수행했습니다.")
        print("자동 home 이동은 안전 확인 전까지 실행하지 않습니다.")
        print("먼저 1_5_기구제어검증_모터_방향_테스트.py로 각 축 방향을 확인하세요.")
