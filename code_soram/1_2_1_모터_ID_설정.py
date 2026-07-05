import serial
import time
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "robot_config.json"

def load_serial_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    port = config.get("port")
    baudrate = config.get("baudrate", 1000000)

    if not port:
        raise ValueError("robot_config.json에 'port' 값이 없습니다.")

    return port, baudrate


PORT, BAUDRATE = load_serial_config()

ser = serial.Serial(PORT, BAUDRATE, timeout=1)

def force_permanent_id(old_id, new_id):
    print(f"🔄 ID {old_id} -> {new_id} 영구 변경 시도 중...")
    
    # 1. Feetech 계열에서 사용하는 모든 Lock 주소(48번, 55번)를 안전하게 모두 해제(0)
    for lock_reg in [48, 55]: # 48 = 0x30, 55 = 0x37
        pkt = bytes([0xFF, 0xFF, old_id, 0x04, 0x03, lock_reg, 0x00])
        pkt += bytes([(~sum(pkt[2:])) & 0xFF])
        ser.write(pkt)
        time.sleep(0.02)
    
    # 2. 새 ID 쓰기 (주소 0x05)
    id_pkt = bytes([0xFF, 0xFF, old_id, 0x04, 0x03, 0x05, new_id])
    id_pkt += bytes([(~sum(id_pkt[2:])) & 0xFF])
    ser.write(id_pkt)
    
    # ⭐ 핵심: 세팅 후 다시 잠그지 않고, 메모리에 각인될 때까지 0.2초 동안 전원을 유지하며 대기
    time.sleep(0.2) 
    
    print("✅ 영구 변경 명령 전송 완료!")

# 현재 서보가 다시 1번으로 돌아갔으므로 old_id=1, new_id=2로 설정
force_permanent_id(1, 3)
ser.close()

