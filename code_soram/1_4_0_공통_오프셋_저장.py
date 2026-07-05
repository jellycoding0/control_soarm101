import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from so_servo_bus import CONFIG_PORT, FeetechServoBus


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
JOINT_IDS = [1, 2, 3, 4, 5]
SAMPLES = 10


def read_average_positions(bus, joint_ids, samples):
    totals = {joint_id: 0 for joint_id in joint_ids}
    for _ in range(samples):
        for joint_id in joint_ids:
            totals[joint_id] += bus.read_present_position(joint_id)
        time.sleep(0.03)
    return {joint_id: round(totals[joint_id] / samples) for joint_id in joint_ids}


def positions_to_offsets_deg(bus, positions):
    offsets = []
    for joint_id in JOINT_IDS:
        center_diff = positions[joint_id] - bus.CENTER_POS
        offsets.append(round(center_diff * bus.DEG_PER_STEP, 6))
    return offsets


def main():
    parser = argparse.ArgumentParser(description="현재 자세의 엔코더 값을 오프셋 JSON으로 저장합니다.")
    parser.add_argument("--port", default=CONFIG_PORT, help=f"시리얼 포트. 기본값: {CONFIG_PORT}")
    parser.add_argument("--pose-name", required=True, help="저장할 자세 이름. 예: resting_home, home_vertical")
    parser.add_argument("--output", required=True, help="저장할 JSON 파일 경로")
    parser.add_argument("--samples", type=int, default=SAMPLES, help=f"평균 샘플 수. 기본값: {SAMPLES}")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    bus = FeetechServoBus(port=args.port, simulate=False)
    try:
        bus.open()
        print("기준 자세를 손으로 맞출 수 있도록 토크를 OFF합니다.")
        bus.set_torque([1, 2, 3, 4, 5, 6], enable=False)
        input("\n로봇을 저장할 기준 자세에 맞춘 뒤 Enter를 누르세요.\n> ")

        print("\n엔코더값 평균을 읽는 중...")
        positions = read_average_positions(bus, JOINT_IDS, args.samples)
        offsets = positions_to_offsets_deg(bus, positions)

        data = {
            "pose_name": args.pose_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "port": args.port,
            "joint_ids": JOINT_IDS,
            "present_positions": [positions[joint_id] for joint_id in JOINT_IDS],
            "joint_offsets_deg": offsets,
        }
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n[저장 완료]")
        print(f"파일: {output}")
        print(f"JOINT_OFFSETS_DEG = {offsets}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
