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

ser = serial.Serial(PORT, BAUDRATE, timeout=0.05) # 빠른 스캔을 위해 타임아웃 단축

print("🔍 연결된 서보 ID 스캔을 시작합니다...")
found_any = False

# 1번부터 10번 ID까지 차례대로 핑을 날려봅니다 (필요하면 범위를 253까지 늘려도 됨)
for servo_id in range(1, 11):
    # PING 패킷 생성: FF FF [ID] [LEN=2] [CMD=0x01] [CS]
    pkt = bytes([0xFF, 0xFF, servo_id, 0x02, 0x01])
    pkt += bytes([(~sum(pkt[2:])) & 0xFF]) # 체크섬 계산
    
    # 시리얼 버퍼 비우고 패킷 전송
    ser.reset_input_buffer()
    ser.write(pkt)
    time.sleep(0.01) # 서보가 응답할 때까지 아주 잠깐 대기
    
    # 서보의 응답 패킷 읽기 (기본 구조: FF FF ID LEN ERR CS -> 최소 6바이트)
    response = ser.read(6)
    
    # 응답이 오고, 헤더(FF FF)가 맞으며, 해당 ID가 응답했는지 확인
    if len(response) >= 4 and response[0] == 0xFF and response[1] == 0xFF:
        if response[2] == servo_id:
            print(f"✅ [확인 완료] ID {servo_id}번 서보가 정상 응답했습니다! (ERR 상태 코드: {response[4]})")
            found_any = True
        else:
            print(f"❓ 알 수 없는 응답 (요청 ID: {servo_id} / 응답 ID: {response[2]})")

if not found_any:
    print("❌ 응답하는 서보가 없습니다. 포트 번호, 전원(12V), 또는 배선을 다시 확인하세요.")

print("🏁 스캔 종료!")
ser.close()

