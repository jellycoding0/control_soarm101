import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from so_robot_controller import SoArm101ProController, CONFIG_PORT

PORT = CONFIG_PORT
GRIPPER_LIMITS_DEG = (-60.0, 60.0)
DURATION = 3.0


def main():
    parser = argparse.ArgumentParser(
        description="SO-ARM101 Pro Joint Space Path (Point A) GUI 제어 프로그램"
    )
    parser.add_argument("--port", default=PORT, help=f"시리얼 포트. 기본값: {PORT}")
    parser.add_argument("--simulate", action="store_true", help="시뮬레이션 모드로 실행")
    parser.add_argument("--duration", type=float, default=DURATION, help=f"이동 시간(sec). 기본값: {DURATION}")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than 0")

    print("==================================================")
    print("[ROBOT] SO-ARM101 Pro Joint Space Path GUI 제어기")
    print("  경로 방식: 관절 공간 보간 (TCP 경로 = 곡선)")
    print(f"포트: {args.port} | 모드: {'시뮬레이션' if args.simulate else '실물 장비'}")
    print("==================================================")

    # Initialize Controller
    try:
        arm = SoArm101ProController(port=args.port, simulate=args.simulate, dt=0.02)
        arm.open()
        if args.simulate and hasattr(arm, "fig"):
            plt.close(arm.fig)

        try:
            print("[SYSTEM] 초기 로봇 상태 동기화 및 토크 잠금 중...")
            if not arm.bus.simulate:
                arm.move_joints(arm.current_joints, duration=0.2)
            print("[SYSTEM] 준비 완료!")
        except Exception as e:
            print(f"[WARN] 초기 토크 활성화 실패 (전원이 차단되어 있을 수 있음): {e}")

    except Exception as e:
        print(f"[ERROR] 포트 연결 및 기구부 초기화 실패: {e}")
        sys.exit(1)

    plt.ioff()
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title("SO-ARM101 Pro [Joint Space Path] Point A Controller")

    ax = fig.add_axes([0.05, 0.1, 0.5, 0.8], projection='3d')

    # Calculate initial TCP position
    T_init, _ = arm.kinematics.forward_kinematics(arm.current_joints)
    x_init, y_init, z_init = T_init[0, 3], T_init[1, 3], T_init[2, 3]

    # Target Joint angles (Ghost Robot)
    target_joints = arm.current_joints.copy()

    # Read start gripper position
    if arm.bus.simulate:
        start_gripper_deg = 0.0
    else:
        try:
            start_gripper_deg = np.rad2deg(arm.bus.pos_to_rad(arm.bus.read_present_position(6), 6))
        except Exception:
            start_gripper_deg = 0.0
    arm.current_gripper_deg = start_gripper_deg

    # -------------------------------------------------------------------------
    # Sliders: 목표 XYZ (TaskSpace와 동일하게 IK로 target_joints 계산)
    # -------------------------------------------------------------------------
    ax_x  = fig.add_axes([0.68, 0.80, 0.22, 0.03])
    ax_y  = fig.add_axes([0.68, 0.69, 0.22, 0.03])
    ax_z  = fig.add_axes([0.68, 0.58, 0.22, 0.03])
    ax_j6 = fig.add_axes([0.68, 0.47, 0.22, 0.03])

    slider_x  = Slider(ax_x,  'Target X (mm)', -250.0, 250.0, valinit=x_init, valstep=1.0)
    slider_y  = Slider(ax_y,  'Target Y (mm)', -250.0, 250.0, valinit=y_init, valstep=1.0)
    slider_z  = Slider(ax_z,  'Target Z (mm)',  -50.0, 450.0, valinit=z_init, valstep=1.0)
    slider_j6 = Slider(ax_j6, 'Target Gripper', GRIPPER_LIMITS_DEG[0], GRIPPER_LIMITS_DEG[1], valinit=start_gripper_deg, valstep=1.0)

    sliders = [slider_x, slider_y, slider_z, slider_j6]
    for s in sliders:
        s.label.set_fontsize(9)

    # Buttons
    btn_move_ax      = fig.add_axes([0.68, 0.12, 0.08, 0.04])
    btn_move         = Button(btn_move_ax,      'Move to A')
    btn_home_ax      = fig.add_axes([0.77, 0.12, 0.05, 0.04])
    btn_home         = Button(btn_home_ax,      'Home')
    btn_torque_on_ax = fig.add_axes([0.83, 0.12, 0.07, 0.04])
    btn_torque_on    = Button(btn_torque_on_ax, 'Torque ON')
    btn_torque_off_ax= fig.add_axes([0.91, 0.12, 0.07, 0.04])
    btn_torque_off   = Button(btn_torque_off_ax,'Torque OFF')

    is_moving    = [False]
    is_resetting = [False]

    # -------------------------------------------------------------------------
    # Draw function for 3D simulator
    # -------------------------------------------------------------------------
    def draw_robot():
        ax.cla()

        # 1. Floor grid
        x_grid, y_grid = np.meshgrid(np.linspace(-300, 300, 7), np.linspace(-300, 300, 7))
        z_grid = np.zeros_like(x_grid)
        ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightgray', alpha=0.7, linewidth=0.8)

        _, current_frames = arm.kinematics.forward_kinematics(arm.current_joints)
        _, target_frames  = arm.kinematics.forward_kinematics(target_joints)

        # 2. Base pedestal
        shoulder_height = current_frames[1][2, 3]
        ax.plot([0, 0], [0, 0], [0, shoulder_height], color='dimgray', linewidth=12, solid_capstyle='round')

        # 3. Target Ghost Robot (Dashed Gray)
        for i in range(1, 6):
            p_start = target_frames[i]
            p_end   = target_frames[i + 1]
            ax.plot(
                [p_start[0,3], p_end[0,3]],
                [p_start[1,3], p_end[1,3]],
                [p_start[2,3], p_end[2,3]],
                '--o', color='gray', linewidth=2.5, markersize=5, alpha=0.6, solid_capstyle='round'
            )

        # Ghost Gripper
        tg_u_x   = target_frames[6][:3, 1]
        tg_u_z   = target_frames[6][:3, 2]
        tg_p_base= target_frames[6][:3, 3]
        tg_w_open= 5.0 + 45.0 * (slider_j6.val - GRIPPER_LIMITS_DEG[0]) / (GRIPPER_LIMITS_DEG[1] - GRIPPER_LIMITS_DEG[0])
        tg_w_base= 25.0; tg_l_finger = 35.0
        tg_p_lb  = tg_p_base - (tg_w_base/2.0)*tg_u_x
        tg_p_rb  = tg_p_base + (tg_w_base/2.0)*tg_u_x
        tg_p_lt  = tg_p_base + tg_l_finger*tg_u_z - (tg_w_open/2.0)*tg_u_x
        tg_p_rt  = tg_p_base + tg_l_finger*tg_u_z + (tg_w_open/2.0)*tg_u_x
        ax.plot([tg_p_lb[0],tg_p_rb[0]], [tg_p_lb[1],tg_p_rb[1]], [tg_p_lb[2],tg_p_rb[2]], color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax.plot([tg_p_lb[0],tg_p_lt[0]], [tg_p_lb[1],tg_p_lt[1]], [tg_p_lb[2],tg_p_lt[2]], color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax.plot([tg_p_rb[0],tg_p_rt[0]], [tg_p_rb[1],tg_p_rt[1]], [tg_p_rb[2],tg_p_rt[2]], color='gray', linestyle='--', linewidth=2, alpha=0.5)

        # 4. Current Robot (Solid Colored Links)
        link_colors = ['darkorange', 'limegreen', 'royalblue', 'mediumpurple', 'teal']
        for i in range(1, 6):
            p_start = current_frames[i]
            p_end   = current_frames[i + 1]
            ax.plot(
                [p_start[0,3], p_end[0,3]],
                [p_start[1,3], p_end[1,3]],
                [p_start[2,3], p_end[2,3]],
                '-o', color=link_colors[i-1], linewidth=6, markersize=9, solid_capstyle='round'
            )
        ax.scatter(current_frames[-1][0,3], current_frames[-1][1,3], current_frames[-1][2,3],
                   color='crimson', s=100, label='TCP', zorder=10)

        # Current Gripper
        u_x    = current_frames[6][:3, 1]
        u_z    = current_frames[6][:3, 2]
        p_base = current_frames[6][:3, 3]
        w_open = 5.0 + 45.0 * (arm.current_gripper_deg - GRIPPER_LIMITS_DEG[0]) / (GRIPPER_LIMITS_DEG[1] - GRIPPER_LIMITS_DEG[0])
        w_base = 25.0; l_finger = 35.0
        p_lb = p_base - (w_base/2.0)*u_x;  p_rb = p_base + (w_base/2.0)*u_x
        p_lt = p_base + l_finger*u_z - (w_open/2.0)*u_x
        p_rt = p_base + l_finger*u_z + (w_open/2.0)*u_x
        ax.plot([p_lb[0],p_rb[0]], [p_lb[1],p_rb[1]], [p_lb[2],p_rb[2]], color='darkslategray', linewidth=4)
        ax.plot([p_lb[0],p_lt[0]], [p_lb[1],p_lt[1]], [p_lb[2],p_lt[2]], color='darkslategray', linewidth=4, solid_capstyle='round')
        ax.plot([p_rb[0],p_rt[0]], [p_rb[1],p_rt[1]], [p_rb[2],p_rt[2]], color='darkslategray', linewidth=4, solid_capstyle='round')

        # 5. Joint Labels
        for i, name in enumerate(["J1","J2","J3","J4","J5","TCP"], start=1):
            p = current_frames[i][:3, 3]
            ax.text(p[0]+8, p[1]+8, p[2]+8, name, color='black', fontsize=9, fontweight='bold')

        ax.set_xlim([-300, 300]); ax.set_ylim([-300, 300]); ax.set_zlim([-20, 500])
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title("Joint Space Path - Joint Space Interpolation (TCP Path = Curve)")

        curr_deg = np.rad2deg(arm.current_joints)
        targ_deg = np.rad2deg(target_joints)
        hud_info = (
            f"[Joint Space Path] Direct Joint Angle Interpolation\n"
            f"TCP Path: Curve (Linear Joint Angle Interpolation)\n"
            f"-----------------------------------\n"
            f"Joint | Current | Target A\n"
            f" J1   | {curr_deg[0]:6.1f} deg | {targ_deg[0]:6.1f} deg\n"
            f" J2   | {curr_deg[1]:6.1f} deg | {targ_deg[1]:6.1f} deg\n"
            f" J3   | {curr_deg[2]:6.1f} deg | {targ_deg[2]:6.1f} deg\n"
            f" J4   | {curr_deg[3]:6.1f} deg | {targ_deg[3]:6.1f} deg\n"
            f" J5   | {curr_deg[4]:6.1f} deg | {targ_deg[4]:6.1f} deg\n"
            f"-----------------------------------\n"
            f"Target XYZ: [{slider_x.val:.0f}, {slider_y.val:.0f}, {slider_z.val:.0f}] mm"
        )
        ax.text2D(0.02, 0.98, hud_info, transform=ax.transAxes, fontsize=9,
                  family='monospace', verticalalignment='top',
                  bbox=dict(facecolor='whitesmoke', alpha=0.9, boxstyle='round,pad=0.4', edgecolor='silver'))

        fig.canvas.draw_idle()

    draw_robot()

    # -------------------------------------------------------------------------
    # Slider callback: IK로 target_joints 계산 (목표 설정 방식은 TaskSpace와 동일)
    # -------------------------------------------------------------------------
    def on_slider_changed(val):
        nonlocal target_joints
        if is_resetting[0] or is_moving[0]:
            return

        target_xyz = np.array([slider_x.val, slider_y.val, slider_z.val])
        q_target = arm.kinematics.inverse_kinematics_position_dls(target_xyz, target_joints)

        try:
            arm._check_joint_limits(q_target)
            target_joints = q_target
        except ValueError:
            pass

        draw_robot()

    slider_x.on_changed(on_slider_changed)
    slider_y.on_changed(on_slider_changed)
    slider_z.on_changed(on_slider_changed)
    slider_j6.on_changed(on_slider_changed)

    # -------------------------------------------------------------------------
    # Trajectory execution: 관절 공간 보간 (q0 → q1 선형 보간)
    # TCP는 곡선 경로를 따라 이동
    # -------------------------------------------------------------------------
    def move_to_a_trajectory():
        if is_moving[0]:
            return
        is_moving[0] = True
        print("\n[MOVE] [Joint Space Path] 관절각도 직접 보간으로 이동 시작")
        print("       → TCP는 곡선 경로를 따라 이동합니다.")

        q0 = arm.current_joints.copy()
        q1 = target_joints.copy()   # IK로 계산된 목표 관절각도
        g0 = arm.current_gripper_deg
        g1 = slider_j6.val

        duration = args.duration
        steps    = max(1, int(np.ceil(duration / arm.dt)))
        start_time = time.time()

        for i in range(steps + 1):
            t = min(i * arm.dt, duration)
            s = (1.0 - np.cos(np.pi * t / duration)) / 2.0  # Cosine S-curve

            # ★ 관절 공간 보간: 관절각도를 직접 선형 보간
            q_curr = q0 + s * (q1 - q0)
            g_curr = g0 + s * (g1 - g0)

            arm.current_joints      = q_curr
            arm.current_gripper_deg = g_curr

            if not arm.bus.simulate:
                try:
                    arm.bus.sync_write_positions(arm.joint_ids, q_curr)
                    gripper_pos = int(2048 + g_curr / (360.0 / 4096.0))
                    arm.bus.write_goal_position_raw(6, gripper_pos)
                except Exception as e:
                    print(f"[ERROR] 구동 중 통신 오류: {e}")

            if i % 5 == 0 or i == steps:
                draw_robot()
                plt.pause(0.001)

            elapsed    = time.time() - start_time
            sleep_time = (i + 1) * arm.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[MOVE] 목표 지점 A에 도달했습니다.")
        is_moving[0] = False

    def on_move_to_a(event):
        move_to_a_trajectory()

    def on_home(event):
        nonlocal target_joints
        if is_moving[0]:
            return
        print("[MOVE] 홈 포지션으로 복귀합니다.")
        is_resetting[0] = True

        T_home, _ = arm.kinematics.forward_kinematics(np.zeros(5))
        x_h, y_h, z_h = T_home[0,3], T_home[1,3], T_home[2,3]

        for s in sliders:
            s.eventson = False
        slider_x.set_val(x_h); slider_y.set_val(y_h); slider_z.set_val(z_h)
        slider_j6.set_val(0.0)
        for s in sliders:
            s.eventson = True

        target_joints    = np.zeros(5)
        is_resetting[0]  = False
        move_to_a_trajectory()

    def on_torque_on(event):
        if not arm.bus.simulate:
            try:
                arm.bus.set_torque(arm.all_ids, enable=True)
                print("[TORQUE] 모든 모터의 토크를 켰습니다.")
                arm.motion_ready = True
            except Exception as e:
                print(f"[WARN] 토크 활성화 실패: {e}")

    def on_torque_off(event):
        if not arm.bus.simulate:
            try:
                arm.bus.set_torque(arm.all_ids, enable=False)
                print("[TORQUE] 모든 모터의 토크를 해제했습니다.")
                arm.motion_ready = False
            except Exception as e:
                print(f"[WARN] 토크 해제 실패: {e}")

    btn_move.on_clicked(on_move_to_a)
    btn_home.on_clicked(on_home)
    btn_torque_on.on_clicked(on_torque_on)
    btn_torque_off.on_clicked(on_torque_off)

    fig.btn_handles = [btn_move, btn_home, btn_torque_on, btn_torque_off]

    try:
        plt.show()
    finally:
        print("[SYSTEM] 프로그램 종료 중... 안전을 위해 모터 토크를 해제합니다.")
        arm.close()


if __name__ == "__main__":
    main()
