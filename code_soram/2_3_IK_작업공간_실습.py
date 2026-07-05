import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from so_robot_controller import SoArm101ProController, CONFIG_PORT

PORT = CONFIG_PORT
GRIPPER_LIMITS_DEG = (-60.0, 60.0)

# Global variables for rate limiting
last_update_time = [0.0]
UPDATE_INTERVAL_SEC = 0.05  # 20Hz update rate for smooth real-time control


def main():
    parser = argparse.ArgumentParser(
        description="SO-ARM101 Pro Task Space (XYZ) Slider GUI 제어 프로그램"
    )
    parser.add_argument("--port", default=PORT, help=f"시리얼 포트. 기본값: {PORT}")
    parser.add_argument("--simulate", action="store_true", help="시뮬레이션 모드로 실행")
    args = parser.parse_args()

    print("==================================================")
    print("[ROBOT] SO-ARM101 Pro 3D Task Space (XYZ) GUI 제어기")
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
    fig.canvas.manager.set_window_title("SO-ARM101 Pro 3D Task Space (XYZ) Jog Controller")

    # 3D Subplot for Robot representation
    ax = fig.add_axes([0.05, 0.1, 0.5, 0.8], projection='3d')
    
    # Calculate initial TCP position
    T_init, _ = arm.kinematics.forward_kinematics(arm.current_joints)
    x_init, y_init, z_init = T_init[0, 3], T_init[1, 3], T_init[2, 3]

    # Read start gripper position
    if arm.bus.simulate:
        start_gripper_deg = 0.0
    else:
        try:
            start_gripper_deg = np.rad2deg(arm.bus.pos_to_rad(arm.bus.read_present_position(6), 6))
        except Exception as e:
            start_gripper_deg = 0.0
    arm.current_gripper_deg = start_gripper_deg

    # Sliders Subplots on the right side
    ax_x = fig.add_axes([0.68, 0.80, 0.22, 0.03])
    ax_y = fig.add_axes([0.68, 0.69, 0.22, 0.03])
    ax_z = fig.add_axes([0.68, 0.58, 0.22, 0.03])
    ax_j6 = fig.add_axes([0.68, 0.47, 0.22, 0.03])

    slider_x = Slider(ax_x, 'X (mm)', -250.0, 250.0, valinit=x_init, valstep=1.0)
    slider_y = Slider(ax_y, 'Y (mm)', -250.0, 250.0, valinit=y_init, valstep=1.0)
    slider_z = Slider(ax_z, 'Z (mm)', -50.0, 450.0, valinit=z_init, valstep=1.0)
    slider_j6 = Slider(ax_j6, 'Gripper (deg)', GRIPPER_LIMITS_DEG[0], GRIPPER_LIMITS_DEG[1], valinit=start_gripper_deg, valstep=1.0)

    # Set Slider Colors
    slider_x.label.set_fontsize(10)
    slider_y.label.set_fontsize(10)
    slider_z.label.set_fontsize(10)
    slider_j6.label.set_fontsize(10)

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
        
        # 4. Draw Gripper fingers
        u_x = frames[6][:3, 1]  # Sideways direction of the fingers (Y-axis for 90 deg rotation)
        u_z = frames[6][:3, 2]  # Forward direction of the fingers
        p_base = frames[6][:3, 3]  # Base of the gripper
        
        # Map current_gripper_deg to opening width (5mm to 50mm)
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
        ax.set_title("SO-ARM101 Pro 3D Simulation (Task Space)")

        # Update HUD text display showing joint angles (in degrees)
        q_deg = np.rad2deg(arm.current_joints)
        hud_info = (
            f"Joint Angles (Model)\n"
            f"---------------------------\n"
            f"J1: {q_deg[0]:6.1f}°\n"
            f"J2: {q_deg[1]:6.1f}°\n"
            f"J3: {q_deg[2]:6.1f}°\n"
            f"J4: {q_deg[3]:6.1f}°\n"
            f"J5: {q_deg[4]:6.1f}°"
        )
        
        ax.text2D(
            0.02, 0.98, hud_info,
            transform=ax.transAxes,
            fontsize=10,
            family='monospace',
            verticalalignment='top',
            bbox=dict(facecolor='whitesmoke', alpha=0.85, boxstyle='round,pad=0.4', edgecolor='silver')
        )

        fig.canvas.draw_idle()

    # Initial Draw
    draw_robot()

    # Flag to prevent callback loops during reset
    is_resetting = [False]

    # Callback for slider updates
    def on_slider_changed(val):
        if is_resetting[0]:
            return
            
        target_xyz = np.array([slider_x.val, slider_y.val, slider_z.val])
        arm.current_gripper_deg = slider_j6.val
        
        # Run Inverse Kinematics (DLS)
        q_target = arm.kinematics.inverse_kinematics_position_dls(target_xyz, arm.current_joints)
        
        # Check joint limits before applying
        try:
            arm._check_joint_limits(q_target)
            arm.current_joints = q_target
        except ValueError:
            pass  # Ignore target step if limits are exceeded to block movement safely
            
        # Redraw 3D simulator in real-time
        draw_robot()
        
        # Rate-limited write to the physical servos
        curr_time = time.time()
        if curr_time - last_update_time[0] > UPDATE_INTERVAL_SEC:
            if not arm.bus.simulate:
                try:
                    arm.bus.sync_write_positions(arm.joint_ids, arm.current_joints)
                    # Gripper
                    gripper_pos = int(2048 + slider_j6.val / (360.0 / 4096.0))
                    arm.bus.write_goal_position_raw(6, gripper_pos)
                except Exception as e:
                    print(f"[WARN] 통신 중 오류: {e}")
            last_update_time[0] = curr_time

    # Mouse release handler to ensure final position accuracy
    def on_release(event):
        if is_resetting[0]:
            return
            
        if not arm.bus.simulate:
            try:
                arm.bus.sync_write_positions(arm.joint_ids, arm.current_joints)
                gripper_pos = int(2048 + slider_j6.val / (360.0 / 4096.0))
                arm.bus.write_goal_position_raw(6, gripper_pos)
            except Exception as e:
                print(f"[WARN] 통신 중 오류: {e}")

    # Register slider events
    slider_x.on_changed(on_slider_changed)
    slider_y.on_changed(on_slider_changed)
    slider_z.on_changed(on_slider_changed)
    slider_j6.on_changed(on_slider_changed)
    
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Button handlers
    def on_home(event):
        print("[HOME] 홈 포지션으로 이동합니다.")
        is_resetting[0] = True
        arm.home()
        
        # Reset sliders to home position (which corresponds to q=all zero)
        T_home, _ = arm.kinematics.forward_kinematics(arm.current_joints)
        x_h, y_h, z_h = T_home[0, 3], T_home[1, 3], T_home[2, 3]
        
        slider_x.set_val(x_h)
        slider_y.set_val(y_h)
        slider_z.set_val(z_h)
        slider_j6.set_val(0.0)
        is_resetting[0] = False
        
        draw_robot()

    def on_torque_on(event):
        if not arm.bus.simulate:
            try:
                arm.bus.set_torque(arm.all_ids, enable=True)
                print("[POWER] 모든 모터의 토크를 켰습니다 (움직임 가능).")
                arm.motion_ready = True
            except Exception as e:
                print(f"[WARN] 토크 활성화 실패: {e}")

    def on_torque_off(event):
        if not arm.bus.simulate:
            try:
                arm.bus.set_torque(arm.all_ids, enable=False)
                print("[LOCK] 모든 모터의 토크를 해제했습니다 (흐물흐물 상태).")
                arm.motion_ready = False
            except Exception as e:
                print(f"[WARN] 토크 해제 실패: {e}")

    btn_home.on_clicked(on_home)
    btn_torque_on.on_clicked(on_torque_on)
    btn_torque_off.on_clicked(on_torque_off)

    # Prevent garbage collection of Matplotlib widget connections
    fig.btn_handles = [btn_home, btn_torque_on, btn_torque_off]

    try:
        plt.show()
    finally:
        # Cleanup when window is closed to release torque safely
        print("[PORT] 프로그램 종료 중... 안전을 위해 모터 토크를 해제합니다.")
        arm.close()


if __name__ == "__main__":
    main()
