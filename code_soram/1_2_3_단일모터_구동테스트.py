import serial
import time
import numpy as np

# 윈도우 장치 관리자에서 확인하신 포트 고정
PORT = 'COM11'  
BAUDRATE = 1000000

CENTER_POS = 2048
DEG_PER_STEP = 360.0 / 4096.0

def set_torque(ser, servo_id, enable):
    val = 1 if enable else 0
    # 데이터가 1바이트이므로 LEN = 4 (정상)
    pkt = [0xFF, 0xFF, servo_id, 0x04, 0x03, 0x28, val]
    pkt.append((~sum(pkt[2:])) & 0xFF)
    ser.write(bytes(pkt))
    time.sleep(0.01)

def send_position(ser, servo_id, deg):
    pos = int(CENTER_POS + deg / DEG_PER_STEP)
    pos = max(0, min(4095, pos))
    
    # ⭐ 수정완료: 주소(1B) + 데이터(2B) = 파라미터 3B. 따라서 LEN = 3 + 2 = 0x05!!
    pkt = [0xFF, 0xFF, servo_id, 0x05, 0x03, 0x2A, pos & 0xFF, (pos >> 8) & 0xFF]
    pkt.append((~sum(pkt[2:])) & 0xFF)
    ser.write(bytes(pkt))

def move_servo_slow(ser, servo_id, start_deg, end_deg, duration=1.5):
    dt = 0.02
    steps = int(duration / dt)
    for i in range(steps + 1):
        t = i * dt
        s = (1.0 - np.cos(np.pi * t / duration)) / 2.0
        curr_deg = start_deg + s * (end_deg - start_deg)
        send_position(ser, servo_id, curr_deg)
        time.sleep(dt)

def main():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
        print(f"🔌 포트 연결 성공: {PORT}")
    except Exception as e:
        print(f"❌ 포트 연결 실패: {e}")
        return

    print("⚙️ 1~6번 모터 토크 ON! (징~ 소리가 나며 잠깁니다)")
    for i in range(1, 7):
        set_torque(ser, i, True)
    time.sleep(0.5)

    print("\n🚀 [시원하게 90도 회전 테스트 시작]")
    for servo_id in range(1, 7):
        print(f"▶️ ID {servo_id}번 모터: 90도 회전 중...")
        # 0도 -> 90도로 천천히 이동 (1.5초)
        move_servo_slow(ser, servo_id, 0, 90, duration=1.5)
        time.sleep(0.2)
        # 90도 -> 0도로 천천히 복귀 (1.5초)
        move_servo_slow(ser, servo_id, 90, 0, duration=1.5)
        time.sleep(0.5)

    print("\n🏁 테스트 완료! 안전을 위해 토크를 해제합니다.")
    for i in range(1, 7):
        set_torque(ser, i, False)
    ser.close()

if __name__ == "__main__":
    main()
