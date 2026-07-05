import argparse
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from so_robot_controller import CONFIG_PORT, SoArm101ProController, JOINT_LIMITS_DEG

PORT = CONFIG_PORT
GRIPPER_LIMITS_DEG = (-60.0, 60.0)

# Global variables for rate limiting
last_update_time = [0.0]
UPDATE_INTERVAL_SEC = 0.05  # 20Hz update rate for smooth real-time control


def get_euler_angles(R):
    # Robust Roll-Pitch-Yaw extraction handling gimbal lock
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)


def main():
    parser = argparse.ArgumentParser(
        description="SO-ARM101 Pro Slider GUI 제어 프로그램"
    )
    parser.add_argument("--port", default=PORT, help=f"시리얼 포트. 기본값: {PORT}")
    parser.add_argument("--simulate", action="store_true", help="시뮬레이션 모드로 실행")
    args = parser.parse_args()

    print("==================================================")
    print("[ROBOT] SO-ARM101 Pro 3D GUI & Slider (J1-6) 제어기")
    print(f"포트: {args.port} | 모드: {'시뮬레이션' if args.simulate else '실물 장비'}")
    print("==================================================")

    # Initialize Controller
    try:
        arm = SoArm101ProController(port=args.port, simulate=args.simulate, dt=0.02)
        arm.open()
        
        # 0-delta initialization to enable torque and set motion_ready=True
        try:
            print("[SYSTEM] 초기 로봇 상태 동기화 및 토크 잠금 중...")
            arm.move_joints(arm.current_joints, duration=0.2)
            print("[SYSTEM] 준비 완료!")
        except Exception as e:
            print(f"[WARN] 초기 토크 활성화 실패 (전원이 차단되어 있을 수 있음): {e}")

    except Exception as e:
        print(f"[ERROR] 포트 연결 및 기구부 초기화 실패: {e}")
        sys.exit(1)

    # Set up Matplotlib Figure
    plt.ioff()  # Turn off interactive mode since we block with plt.show()
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title("SO-ARM101 Pro 3D Simulator & Jog Controller")

    # 3D Subplot for Robot representation
    ax = fig.add_axes([0.05, 0.1, 0.5, 0.8], projection='3d')
    
    # Sliders Subplots on the right side
    slider_axes = [fig.add_axes([0.68, 0.8 - i * 0.11, 0.22, 0.03]) for i in range(6)]
    
    # Read start joints
    start_q_deg = np.rad2deg(arm.current_joints)
    
    # Read start gripper position
    if arm.bus.simulate:
        start_gripper_deg = 0.0
    else:
        try:
            gripper_pos = arm.bus.read_present_position(6)
            start_gripper_deg = (gripper_pos - 2048) * (360.0 / 4096.0)
            start_gripper_deg = max(GRIPPER_LIMITS_DEG[0], min(GRIPPER_LIMITS_DEG[1], start_gripper_deg))
        except:
            start_gripper_deg = 0.0

    # Create Sliders
    slider_j1 = Slider(slider_axes[0], 'J1 Base', JOINT_LIMITS_DEG[0][0], JOINT_LIMITS_DEG[0][1], valinit=start_q_deg[0], valfmt='%1.1f°')
    slider_j2 = Slider(slider_axes[1], 'J2 Shoulder', JOINT_LIMITS_DEG[1][0], JOINT_LIMITS_DEG[1][1], valinit=start_q_deg[1], valfmt='%1.1f°')
    slider_j3 = Slider(slider_axes[2], 'J3 Elbow', JOINT_LIMITS_DEG[2][0], JOINT_LIMITS_DEG[2][1], valinit=start_q_deg[2], valfmt='%1.1f°')
    slider_j4 = Slider(slider_axes[3], 'J4 Wrist Pitch', JOINT_LIMITS_DEG[3][0], JOINT_LIMITS_DEG[3][1], valinit=start_q_deg[3], valfmt='%1.1f°')
    slider_j5 = Slider(slider_axes[4], 'J5 Wrist Roll', JOINT_LIMITS_DEG[4][0], JOINT_LIMITS_DEG[4][1], valinit=start_q_deg[4], valfmt='%1.1f°')
    slider_j6 = Slider(slider_axes[5], 'J6 Gripper', GRIPPER_LIMITS_DEG[0], GRIPPER_LIMITS_DEG[1], valinit=start_gripper_deg, valfmt='%1.1f°')

    sliders = [slider_j1, slider_j2, slider_j3, slider_j4, slider_j5, slider_j6]

    # Create Buttons
    btn_home_ax = fig.add_axes([0.68, 0.1, 0.06, 0.04])
    btn_home = Button(btn_home_ax, 'Home')

    btn_torque_on_ax = fig.add_axes([0.76, 0.1, 0.06, 0.04])
    btn_torque_on = Button(btn_torque_on_ax, 'Torque ON')

    btn_torque_off_ax = fig.add_axes([0.84, 0.1, 0.06, 0.04])
    btn_torque_off = Button(btn_torque_off_ax, 'Torque OFF')

    # Draw function for 3D simulator
    def draw_robot():
        ax.cla()
        _, frames = arm.kinematics.forward_kinematics(arm.current_joints)
        
        # 1. Draw floor grid (Z = 0) representing the work table
        x_grid, y_grid = np.meshgrid(np.linspace(-300, 300, 7), np.linspace(-300, 300, 7))
        z_grid = np.zeros_like(x_grid)
        ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightgray', alpha=0.7, linewidth=0.8)
        
        # 2. Draw base pedestal (Z = 0 to Shoulder Height)
        shoulder_height = frames[1][2, 3]
        ax.plot([0, 0], [0, 0], [0, shoulder_height], color='dimgray', linewidth=12, solid_capstyle='round', label='Base Pedestal')
        
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
            ax.plot(
                [p_start[0, 3], p_end[0, 3]],
                [p_start[1, 3], p_end[1, 3]],
                [p_start[2, 3], p_end[2, 3]],
                '-o',
                color=link_colors[i - 1],
                linewidth=6,
                markersize=9,
                solid_capstyle='round'
            )
            
        # Highlight End-Effector
        ax.scatter(frames[-1][0, 3], frames[-1][1, 3], frames[-1][2, 3], color='crimson', s=120, label='TCP', zorder=10)
        
        u_x = frames[6][:3, 1] # Sideways direction of the fingers (Y-axis for 90 deg rotation)
        u_z = frames[6][:3, 2] # Forward direction of the fingers
        p_base = frames[6][:3, 3] # Base of the gripper
        
        # Map J6 angle to opening width (5mm to 50mm)
        w_open = 5.0 + 45.0 * (slider_j6.val - GRIPPER_LIMITS_DEG[0]) / (GRIPPER_LIMITS_DEG[1] - GRIPPER_LIMITS_DEG[0])
        w_base = 25.0
        l_finger = 35.0
        
        p_left_base = p_base - (w_base / 2.0) * u_x
        p_right_base = p_base + (w_base / 2.0) * u_x
        p_left_tip = p_base + l_finger * u_z - (w_open / 2.0) * u_x
        p_right_tip = p_base + l_finger * u_z + (w_open / 2.0) * u_x
        
        # Draw gripper yoke
        ax.plot(
            [p_left_base[0], p_right_base[0]],
            [p_left_base[1], p_right_base[1]],
            [p_left_base[2], p_right_base[2]],
            color='darkslategray',
            linewidth=4
        )
        # Draw left finger
        ax.plot(
            [p_left_base[0], p_left_tip[0]],
            [p_left_base[1], p_left_tip[1]],
            [p_left_base[2], p_left_tip[2]],
            color='darkslategray',
            linewidth=4,
            solid_capstyle='round'
        )
        # Draw right finger
        ax.plot(
            [p_right_base[0], p_right_tip[0]],
            [p_right_base[1], p_right_tip[1]],
            [p_right_base[2], p_right_tip[2]],
            color='darkslategray',
            linewidth=4,
            solid_capstyle='round'
        )
        
        # 5. Add text labels to each joint axis index
        joint_names = ["J1", "J2", "J3", "J4", "J5", "TCP"]
        for i in range(1, 7):
            p = frames[i][:3, 3]
            ax.text(
                p[0] + 8,
                p[1] + 8,
                p[2] + 8,
                joint_names[i - 1],
                color='black',
                fontsize=9,
                fontweight='bold'
            )
        
        ax.set_xlim([-300, 300])
        ax.set_ylim([-300, 300])
        ax.set_zlim([-20, 500])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title("SO-ARM101 Pro 3D Simulation")

        # Update TCP text display (HUD style in the top-left of 3D axes)
        T_total = frames[-1]
        R = T_total[:3, :3]
        rx, ry, rz = get_euler_angles(R)
        
        tcp_info = (
            f"TCP Pose (End-Effector)\n"
            f"---------------------------\n"
            f"X : {T_total[0, 3]:6.1f} mm\n"
            f"Y : {T_total[1, 3]:6.1f} mm\n"
            f"Z : {T_total[2, 3]:6.1f} mm\n"
            f"rx: {rx:6.1f}°\n"
            f"ry: {ry:6.1f}°\n"
            f"rz: {rz:6.1f}°"
        )
        
        ax.text2D(
            0.02, 0.98, tcp_info,
            transform=ax.transAxes,
            fontsize=9,
            family='monospace',
            verticalalignment='top',
            bbox=dict(facecolor='whitesmoke', alpha=0.85, boxstyle='round,pad=0.4', edgecolor='silver')
        )

        fig.canvas.draw_idle()

    # Initial Draw
    draw_robot()

    # Callback for slider updates
    def on_slider_changed(val):
        # Update current joint angles representation for the 3D plot
        q = np.array([
            np.deg2rad(slider_j1.val),
            np.deg2rad(slider_j2.val),
            np.deg2rad(slider_j3.val),
            np.deg2rad(slider_j4.val),
            np.deg2rad(slider_j5.val),
        ])
        arm.current_joints = q
        
        # Redraw 3D simulator in real-time
        draw_robot()
        
        # Rate-limited write to the physical servos
        curr_time = time.time()
        if curr_time - last_update_time[0] > UPDATE_INTERVAL_SEC:
            if not arm.bus.simulate:
                try:
                    # Direct raw joint positioning for immediate response
                    arm.bus.sync_write_positions(arm.joint_ids, q)
                    
                    # Direct raw gripper position
                    gripper_pos = int(2048 + slider_j6.val / (360.0 / 4096.0))
                    arm.bus.write_goal_position_raw(6, gripper_pos)
                except Exception as e:
                    print(f"[WARN] 통신 중 오류: {e}")
            last_update_time[0] = curr_time

    # Mouse release handler to ensure final position accuracy
    def on_release(event):
        q = np.array([
            np.deg2rad(slider_j1.val),
            np.deg2rad(slider_j2.val),
            np.deg2rad(slider_j3.val),
            np.deg2rad(slider_j4.val),
            np.deg2rad(slider_j5.val),
        ])
        if not arm.bus.simulate:
            try:
                arm.bus.sync_write_positions(arm.joint_ids, q)
                gripper_pos = int(2048 + slider_j6.val / (360.0 / 4096.0))
                arm.bus.write_goal_position_raw(6, gripper_pos)
            except Exception as e:
                print(f"[WARN] 통신 중 오류: {e}")

    # Register slider events
    for slider in sliders:
        slider.on_changed(on_slider_changed)
    
    # Register release event to guarantee final state alignment
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Button Event handlers
    def go_home(event):
        print("[HOME] 홈 자세(0도)로 복귀합니다.")
        # Disconnect callbacks temporarily to prevent event loops
        for s in sliders:
            s.eventson = False
            
        # Reset slider values in GUI
        slider_j1.set_val(0.0)
        slider_j2.set_val(0.0)
        slider_j3.set_val(0.0)
        slider_j4.set_val(0.0)
        slider_j5.set_val(0.0)
        slider_j6.set_val(0.0)
        
        # Re-enable callbacks
        for s in sliders:
            s.eventson = True
            
        # Command robot to Home position
        try:
            arm.home()
            move_gripper_smooth(arm, 0.0, duration=1.0)
            draw_robot()
        except Exception as e:
            print(f"[ERROR] 홈 복귀 실패: {e}")

    def torque_on(event):
        if not arm.bus.simulate:
            arm.bus.set_torque(arm.all_ids, enable=True)
            print("[SYSTEM] 모든 모터 토크 ON (잠금)")
        else:
            print("[시뮬레이션] 모든 모터 토크 ON")

    def torque_off(event):
        if not arm.bus.simulate:
            arm.bus.set_torque(arm.all_ids, enable=False)
            print("[SYSTEM] 모든 모터 토크 OFF (풀림)")
        else:
            print("[시뮬레이션] 모든 모터 토크 OFF")

    btn_home.on_clicked(go_home)
    btn_torque_on.on_clicked(torque_on)
    btn_torque_off.on_clicked(torque_off)

    # Run the GUI block
    print("\n️  GUI 창이 열렸습니다. 슬라이더를 마우스로 조절하여 제어하세요.")
    print("[ERROR] GUI 창을 닫으면 안전하게 프로그램이 종료되며 토크가 해제됩니다.")
    
    try:
        plt.show()
    finally:
        # Cleanup when window is closed
        print("[PORT] 프로그램 종료 중... 안전을 위해 모터 토크를 해제합니다.")
        arm.close()


if __name__ == "__main__":
    main()
