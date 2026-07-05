import time
import json
import serial
import numpy as np
from pathlib import Path

DEFAULT_JOINT_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_PORT = "COM10"
DEFAULT_BAUDRATE = 1000000
DEFAULT_JOINT_DIRECTIONS = [1, 1, 1, 1, 1]
MAX_FIRST_MOVE_STEP = 80  # 약 7도. 이보다 큰 첫 이동은 위험하므로 중단
JOINT_LIMITS_DEG = [
    (-150.0, 150.0),  # J1 Base
    (-90.0, 90.0),    # J2 Shoulder
    (-120.0, 120.0),  # J3 Elbow
    (-90.0, 90.0),    # J4 Wrist Pitch
    (-180.0, 180.0),  # J5 Wrist Roll
]
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
ROBOT_CONFIG_FILE = CONFIG_DIR / "robot_config.json"
RESTING_CALIBRATION_FILE = CONFIG_DIR / "joint_offsets_resting.json"
HOME_VERTICAL_CALIBRATION_FILE = CONFIG_DIR / "joint_offsets_home_vertical.json"


def normalize_deg(deg):
    return ((float(deg) + 180.0) % 360.0) - 180.0


def load_robot_config():
    config = {
        "port": DEFAULT_PORT,
        "baudrate": DEFAULT_BAUDRATE,
        "joint_directions": DEFAULT_JOINT_DIRECTIONS.copy(),
    }
    if not ROBOT_CONFIG_FILE.exists():
        return config

    with ROBOT_CONFIG_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "port" in data:
        config["port"] = str(data["port"])
    if "baudrate" in data:
        config["baudrate"] = int(data["baudrate"])
    if "joint_directions" in data:
        directions = data["joint_directions"]
        if not isinstance(directions, list) or len(directions) != 5:
            raise ValueError(f"joint_directions는 길이 5 리스트여야 합니다: {ROBOT_CONFIG_FILE}")
        directions = [int(x) for x in directions]
        invalid = [x for x in directions if x not in (-1, 1)]
        if invalid:
            raise ValueError(f"joint_directions 값은 1 또는 -1만 가능합니다: {ROBOT_CONFIG_FILE}")
        config["joint_directions"] = directions

    return config


ROBOT_CONFIG = load_robot_config()
CONFIG_PORT = ROBOT_CONFIG["port"]
CONFIG_BAUDRATE = ROBOT_CONFIG["baudrate"]
JOINT_DIRECTIONS = ROBOT_CONFIG["joint_directions"]


def load_joint_offsets_deg(config_path=RESTING_CALIBRATION_FILE, label="캘리브레이션"):
    if not config_path.exists():
        print(f"{label} 파일 없음: {config_path}")
        print("기본 오프셋 [0, 0, 0, 0, 0]을 사용합니다.")
        return DEFAULT_JOINT_OFFSETS_DEG.copy()

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    offsets = data.get("joint_offsets_deg")
    if not isinstance(offsets, list) or len(offsets) != 5:
        raise ValueError(f"joint_offsets_deg는 길이 5 리스트여야 합니다: {config_path}")

    offsets = [float(x) for x in offsets]
    print(f"{label} 오프셋 로드: {offsets}")
    return offsets


def load_kinematic_zero_offsets_deg():
    resting = load_joint_offsets_deg(RESTING_CALIBRATION_FILE, "resting")
    vertical = load_joint_offsets_deg(HOME_VERTICAL_CALIBRATION_FILE, "home_vertical")
    offsets = [normalize_deg((v - r) / d) for v, r, d in zip(vertical, resting, JOINT_DIRECTIONS)]
    print(f"FK/IK 기준 오프셋 계산(home_vertical - resting): {offsets}")
    return offsets


class FeetechServoBus:
    def __init__(self, port=None, baudrate=None, simulate=False, joint_offsets_deg=None, joint_directions=None):
        self.port = CONFIG_PORT if port is None else port
        self.baudrate = CONFIG_BAUDRATE if baudrate is None else baudrate
        self.simulate = simulate
        self.ser = None
        self.joint_offsets_deg = list(load_joint_offsets_deg() if joint_offsets_deg is None else joint_offsets_deg)
        self.joint_directions = list(JOINT_DIRECTIONS if joint_directions is None else joint_directions)
        
        self.CMD_PING = 0x01
        self.CMD_READ = 0x02
        self.CMD_WRITE = 0x03
        self.CMD_SYNC_WRITE = 0x83
        
        self.REG_TORQUE_ENABLE = 0x28
        self.REG_GOAL_POSITION = 0x2A
        self.REG_GOAL_TIME = 0x2C
        self.REG_GOAL_SPEED = 0x2E
        self.REG_PRESENT_POSITION = 0x38
        self.REG_PRESENT_LOAD = 0x3C
        
        self.CENTER_POS = 2048
        self.DEG_PER_STEP = 360.0 / 4096.0
        
    def open(self):
        if not self.simulate:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
                print(f"[PORT] Serial 포트 연결 성공: {self.port}")
            except Exception as e:
                print(f"[ERROR] 포트 연결 실패 ({e}). 시뮬레이션 모드로 강제 전환합니다.")
                self.simulate = True

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[PORT] Serial 포트 연결 해제")

    def set_torque(self, joint_ids, enable=True):
        if self.simulate:
            return
        val = 1 if enable else 0
        for j_id in joint_ids:
            pkt = [0xFF, 0xFF, j_id, 0x04, self.CMD_WRITE, self.REG_TORQUE_ENABLE, val]
            pkt.append((~sum(pkt[2:])) & 0xFF)
            self.ser.write(bytes(pkt))
            time.sleep(0.01)
        print(f"[SYSTEM] 모터 토크 제어 완료 (ID {joint_ids} -> {'ON' if enable else 'OFF'})")

    def get_joint_offset_deg(self, joint_id):
        if 1 <= joint_id <= len(self.joint_offsets_deg):
            return self.joint_offsets_deg[joint_id - 1]
        return 0.0

    def get_joint_direction(self, joint_id):
        if 1 <= joint_id <= len(self.joint_directions):
            return self.joint_directions[joint_id - 1]
        return 1

    def rad_to_pos(self, rad_angle, joint_id=None):
        deg = np.rad2deg(rad_angle)
        if joint_id is not None:
            deg = self.get_joint_direction(joint_id) * deg + self.get_joint_offset_deg(joint_id)
        pos = int(self.CENTER_POS + deg / self.DEG_PER_STEP)
        return max(0, min(4095, pos))

    def pos_to_rad(self, pos, joint_id=None):
        deg = (pos - self.CENTER_POS) * self.DEG_PER_STEP
        if joint_id is not None:
            deg = (deg - self.get_joint_offset_deg(joint_id)) / self.get_joint_direction(joint_id)
        return np.deg2rad(deg)

    def read_present_position(self, servo_id):
        if self.simulate:
            return self.CENTER_POS

        pkt = [
            0xFF,
            0xFF,
            servo_id,
            0x04,
            self.CMD_READ,
            self.REG_PRESENT_POSITION,
            0x02,
        ]
        pkt.append((~sum(pkt[2:])) & 0xFF)

        self.ser.reset_input_buffer()
        self.ser.write(bytes(pkt))
        time.sleep(0.01)

        response = self.ser.read(8)
        if len(response) != 8:
            raise RuntimeError(f"ID {servo_id}: 응답 길이 오류 ({len(response)}B) - {response.hex(' ')}")
        if response[0] != 0xFF or response[1] != 0xFF:
            raise RuntimeError(f"ID {servo_id}: 헤더 오류 - {response.hex(' ')}")
        if response[2] != servo_id:
            raise RuntimeError(f"ID {servo_id}: 다른 ID 응답 ({response[2]}) - {response.hex(' ')}")
        if response[4] != 0:
            raise RuntimeError(f"ID {servo_id}: 서보 에러 코드 0x{response[4]:02X}")

        return response[5] | (response[6] << 8)

    def read_word(self, servo_id, register):
        if self.simulate:
            return self.CENTER_POS

        pkt = [
            0xFF,
            0xFF,
            servo_id,
            0x04,
            self.CMD_READ,
            register,
            0x02,
        ]
        pkt.append((~sum(pkt[2:])) & 0xFF)

        self.ser.reset_input_buffer()
        self.ser.write(bytes(pkt))
        time.sleep(0.01)

        response = self.ser.read(8)
        if len(response) != 8:
            raise RuntimeError(f"ID {servo_id}: 응답 길이 오류 ({len(response)}B) - {response.hex(' ')}")
        if response[0] != 0xFF or response[1] != 0xFF:
            raise RuntimeError(f"ID {servo_id}: 헤더 오류 - {response.hex(' ')}")
        if response[2] != servo_id:
            raise RuntimeError(f"ID {servo_id}: 다른 ID 응답 ({response[2]}) - {response.hex(' ')}")
        if response[4] != 0:
            raise RuntimeError(f"ID {servo_id}: 서보 에러 코드 0x{response[4]:02X}")

        return response[5] | (response[6] << 8)

    def read_goal_position(self, servo_id):
        return self.read_word(servo_id, self.REG_GOAL_POSITION)

    def sync_write_positions(self, joint_ids, rad_angles):
        if self.simulate:
            return
            
        target_reg = self.REG_GOAL_POSITION
        data_len = 2
        
        param = [target_reg, data_len]
        for j_id, rad in zip(joint_ids, rad_angles):
            pos = self.rad_to_pos(rad, j_id)
            param.append(j_id)
            param.append(pos & 0xFF)
            param.append((pos >> 8) & 0xFF)
            
        length = len(param) + 2
        pkt = [0xFF, 0xFF, 0xFE, length, self.CMD_SYNC_WRITE] + param
        checksum = (~sum(pkt[2:])) & 0xFF
        pkt.append(checksum)
        
        self.ser.write(bytes(pkt))

    def write_goal_position_raw(self, servo_id, position):
        if self.simulate:
            return

        position = int(max(0, min(4095, position)))
        pkt = [
            0xFF,
            0xFF,
            servo_id,
            0x05,
            self.CMD_WRITE,
            self.REG_GOAL_POSITION,
            position & 0xFF,
            (position >> 8) & 0xFF,
        ]
        pkt.append((~sum(pkt[2:])) & 0xFF)
        self.ser.write(bytes(pkt))
