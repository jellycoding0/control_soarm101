import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from so_robot_controller import CONFIG_PORT, FeetechServoBus


PORT = CONFIG_PORT
DELTA_DEG = 4.0
ANGULAR_SPEED_DPS = 2
MIN_DURATION = 3.0
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
MAX_TORQUE_ON_DRIFT_STEP = 30
LOG_EVERY_STEPS = 10

JOINT_NAMES = {
    1: "J1 Base",
    2: "J2 Shoulder",
    3: "J3 Elbow",
    4: "J4 Wrist Pitch",
    5: "J5 Wrist Roll",
    6: "Gripper",
}


def ask_enter(message):
    input(f"\n{message}\nEnter를 누르면 계속합니다. Ctrl+C로 중단하면 토크 OFF 후 종료됩니다.\n> ")


def calc_duration(delta_deg, angular_speed_dps):
    if angular_speed_dps <= 0:
        raise ValueError("angular_speed_dps는 0보다 커야 합니다.")
    return max(MIN_DURATION, abs(delta_deg) / angular_speed_dps)


def get_hold_ids(joint_id):
    if 1 <= joint_id <= 5:
        return [1, 2, 3, 4, 5]
    return [1, 2, 3, 4, 5, joint_id]


def hold_current_positions(bus, hold_ids):
    present_by_id = {}
    print(f"자세 유지 토크 대상: {hold_ids}")

    for hold_id in hold_ids:
        present = bus.read_present_position(hold_id)
        present_by_id[hold_id] = present
        bus.write_goal_position_raw(hold_id, present)
        time.sleep(0.03)

        goal = bus.read_goal_position(hold_id)
        print(f"  ID {hold_id}: present={present}, goal={goal}")
        if abs(goal - present) > 2:
            raise RuntimeError(
                f"ID {hold_id}: goal position 초기화 실패. "
                f"present={present}, goal={goal}. 토크를 켜지 않고 중단합니다."
            )

    for hold_id in hold_ids:
        bus.set_torque([hold_id], enable=True)
        time.sleep(0.08)
        present_after = bus.read_present_position(hold_id)
        goal_after = bus.read_goal_position(hold_id)
        drift = present_after - present_by_id[hold_id]
        print(f"  ID {hold_id}: torque ON 후 present={present_after}, goal={goal_after}, drift={drift}")
        if abs(drift) > MAX_TORQUE_ON_DRIFT_STEP:
            bus.set_torque(hold_ids, enable=False)
            raise RuntimeError(
                f"ID {hold_id}: 토크 ON 직후 위치가 {drift} step 튀었습니다. "
                "기구 하중, 조립 간섭, 서보 설정을 먼저 확인하세요."
            )

    return present_by_id


def make_log_writer(joint_id):
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"motor_direction_j{joint_id}_{stamp}.csv"
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["phase", "step", "t_sec", "id", "present", "goal", "goal_minus_present"])
    print(f"엔코더 로그 저장: {path}")
    return path, f, writer


def log_positions(bus, writer, phase, step, t0, ids):
    elapsed = time.time() - t0
    rows = []
    for servo_id in ids:
        try:
            present = bus.read_present_position(servo_id)
            goal = bus.read_goal_position(servo_id)
            diff = goal - present
        except RuntimeError as e:
            present = ""
            goal = ""
            diff = ""
            writer.writerow([phase, step, f"{elapsed:.3f}", servo_id, "READ_ERROR", "", str(e)])
            continue
        rows.append((servo_id, present, goal, diff))
        writer.writerow([phase, step, f"{elapsed:.3f}", servo_id, present, goal, diff])
    return rows


def print_watch_rows(rows):
    watched = [row for row in rows if row[0] in (2, 3, 4)]
    text = " | ".join(
        f"J{servo_id} p={present} g={goal} d={diff}"
        for servo_id, present, goal, diff in watched
    )
    print(text)


def move_one_servo_delta(bus, joint_id, delta_deg, angular_speed_dps):
    hold_ids = get_hold_ids(joint_id)
    log_path, log_file, log_writer = make_log_writer(joint_id)
    t0 = time.time()
    present_by_id = hold_current_positions(bus, hold_ids)

    start_pos = bus.read_present_position(joint_id)
    delta_step = int(round(delta_deg / bus.DEG_PER_STEP))
    target_pos = max(0, min(4095, start_pos + delta_step))
    duration = calc_duration(delta_deg, angular_speed_dps)
    steps = max(1, int(duration / 0.02))

    print(f"\n{JOINT_NAMES[joint_id]}: {delta_deg:+.1f}도 이동")
    print(f"각속도: {angular_speed_dps:.1f} deg/s, 이동시간: {duration:.2f}초")
    print(f"실제 명령 전송 대상: ID {joint_id} 하나만")
    print(f"자세 유지 대상: ID {hold_ids}")
    print(f"raw position: {start_pos} -> {target_pos} (delta {target_pos - start_pos} step)")
    print("움직이는 방향을 눈으로 확인하세요.")

    try:
        rows = log_positions(bus, log_writer, "after_hold", 0, t0, hold_ids)
        print_watch_rows(rows)

        for i in range(steps + 1):
            s = (1.0 - np.cos(np.pi * i / steps)) / 2.0
            pos = int(round(start_pos + s * (target_pos - start_pos)))
            bus.write_goal_position_raw(joint_id, pos)
            if i % LOG_EVERY_STEPS == 0 or i == steps:
                rows = log_positions(bus, log_writer, "move_plus", i, t0, hold_ids)
                print_watch_rows(rows)
            time.sleep(0.02)
        time.sleep(0.5)

        rows = log_positions(bus, log_writer, "after_plus", 0, t0, hold_ids)
        print_watch_rows(rows)

        ask_enter("방향 확인 완료. 원위치로 돌아갑니다.")
        for i in range(steps + 1):
            s = (1.0 - np.cos(np.pi * i / steps)) / 2.0
            pos = int(round(target_pos + s * (start_pos - target_pos)))
            bus.write_goal_position_raw(joint_id, pos)
            if i % LOG_EVERY_STEPS == 0 or i == steps:
                rows = log_positions(bus, log_writer, "return", i, t0, hold_ids)
                print_watch_rows(rows)
            time.sleep(0.02)
        time.sleep(0.5)

        rows = log_positions(bus, log_writer, "after_return", 0, t0, hold_ids)
        print_watch_rows(rows)
    finally:
        for hold_id, present in present_by_id.items():
            bus.write_goal_position_raw(hold_id, present)
        time.sleep(0.05)
        log_positions(bus, log_writer, "before_torque_off", 0, t0, hold_ids)
        log_file.close()
        print(f"엔코더 로그 저장 완료: {log_path}")
        bus.set_torque(hold_ids, enable=False)


def main():
    parser = argparse.ArgumentParser(
        description="SO-ARM101 Pro 첫 실물 구동용: 각 관절을 아주 작은 각도로 움직여 모터 방향을 확인합니다."
    )
    parser.add_argument("--port", default=PORT, help=f"시리얼 포트. 기본값: {PORT}")
    parser.add_argument("--delta-deg", type=float, default=DELTA_DEG, help=f"테스트 각도. 기본값: {DELTA_DEG}도")
    parser.add_argument(
        "--speed-dps",
        type=float,
        default=ANGULAR_SPEED_DPS,
        help=f"테스트 각속도(deg/s). 기본값: {ANGULAR_SPEED_DPS} deg/s",
    )
    parser.add_argument("--joints", default="1,2,3,4,5,6", help="테스트할 모터 ID 목록. 예: 1,2,3 또는 6")
    parser.add_argument("--negative", action="store_true", help="+방향 확인 후 -방향도 확인합니다.")
    args = parser.parse_args()

    joint_ids = [int(x.strip()) for x in args.joints.split(",") if x.strip()]
    for joint_id in joint_ids:
        if joint_id not in JOINT_NAMES:
            raise ValueError(f"지원하지 않는 관절 ID입니다: {joint_id}")

    print("[작은 각도 모터 방향 테스트]")
    print(f"포트: {args.port}")
    print(f"테스트 각도: {args.delta_deg}도")
    print(f"테스트 각속도: {args.speed_dps} deg/s")
    print(f"예상 이동시간: {calc_duration(args.delta_deg, args.speed_dps):.2f}초")
    print("\n준비:")
    print("1. config/joint_offsets_resting.json과 config/robot_config.json이 최신인지 확인")
    print("2. 로봇 주변에 충돌할 물체가 없는지 확인")
    print("3. 전원 차단이 가능한 상태에서 진행")
    print("4. 이상한 방향으로 가면 즉시 Ctrl+C 또는 전원 차단")

    ask_enter("준비가 끝났으면 시작합니다.")

    bus = FeetechServoBus(port=args.port, simulate=False)
    try:
        bus.open()
        current_deg = []
        for joint_id in joint_ids:
            pos = bus.read_present_position(joint_id)
            current_deg.append(round(float(np.rad2deg(bus.pos_to_rad(pos, joint_id))), 3))
        print(f"\n테스트 ID 현재 관절각(deg): {current_deg}")

        for joint_id in joint_ids:
            ask_enter(f"{JOINT_NAMES[joint_id]} +방향 테스트를 시작합니다.")
            move_one_servo_delta(bus, joint_id, abs(args.delta_deg), args.speed_dps)

            if args.negative:
                ask_enter(f"{JOINT_NAMES[joint_id]} -방향 테스트를 시작합니다.")
                move_one_servo_delta(bus, joint_id, -abs(args.delta_deg), args.speed_dps)
    except KeyboardInterrupt:
        print("\n중단했습니다. 테스트 대상 모터 토크를 OFF 합니다.")
        bus.set_torque(joint_ids, enable=False)
    except RuntimeError as e:
        print(f"\n테스트 중단: {e}")
        bus.set_torque([1, 2, 3, 4, 5, 6], enable=False)
    finally:
        bus.close()

    print("\n모든 작은 각도 방향 테스트 완료.")
    print("방향이 예상과 다르면 해당 관절의 조립 방향, 서보 혼 방향, 오프셋 기준 자세를 다시 확인하세요.")


if __name__ == "__main__":
    main()
