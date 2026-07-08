import serial
import time

# ─────────────────────────────────────────────
# 포트 설정 (본인 환경에 맞게 수정)
# ─────────────────────────────────────────────
PORT     = "COM11"      # 장치 관리자에서 확인
BAUDRATE = 1_000_000    # Feetech STS3215 고정값: 1 Mbps


def calc_checksum(packet_from_id: list) -> int:
    """
    체크섬 계산: ID 바이트부터 마지막 데이터 바이트까지 합산 후 하위 1바이트 반전
    checksum = (~sum(ID, LEN, CMD, ...)) & 0xFF
    """
    return (~sum(packet_from_id)) & 0xFF


def build_ping_packet(servo_id: int) -> bytes:
    """
    PING 패킷 조립
    ┌──────┬──────┬────┬─────┬─────┬──────────┐
    │ 0xFF │ 0xFF │ ID │ LEN │ CMD │ CHECKSUM │
    └──────┴──────┴────┴─────┴─────┴──────────┘
      헤더1  헤더2        2    0x01

    LEN = 남은 바이트 수 (CMD + CHECKSUM = 2)
    CMD = 0x01 (PING)
    """
    body = [servo_id, 0x02, 0x01]          # [ID, LEN, CMD]
    checksum = calc_checksum(body)
    packet = [0xFF, 0xFF] + body + [checksum]

    print(f"\n[TX] PING 패킷 (ID={servo_id})")
    print(f"     헤더   : FF FF")
    print(f"     ID     : {servo_id:02X} ({servo_id})")
    print(f"     LEN    : 02  (CMD + CHECKSUM)")
    print(f"     CMD    : 01  (PING)")
    print(f"     체크섬 : {checksum:02X}  = (~({servo_id:02X}+02+01)) & FF")
    print(f"     전체   : {' '.join(f'{b:02X}' for b in packet)}")

    return bytes(packet)


def parse_ping_response(servo_id: int, response: bytes) -> bool:
    """
    PING 응답 패킷 파싱
    ┌──────┬──────┬────┬─────┬───────┬──────────┐
    │ 0xFF │ 0xFF │ ID │ LEN │ ERROR │ CHECKSUM │
    └──────┴──────┴────┴─────┴───────┴──────────┘

    ERROR 비트:
      bit0 = 전압 에러
      bit2 = 과열 에러
      bit5 = 과부하 에러
    """
    print(f"[RX] 응답: {' '.join(f'{b:02X}' for b in response)}")

    if len(response) < 6:
        print(f"  ✗ 응답 길이 부족 ({len(response)}바이트, 최소 6 필요)")
        return False

    if response[0] != 0xFF or response[1] != 0xFF:
        print(f"  ✗ 헤더 오류: {response[0]:02X} {response[1]:02X}")
        return False

    if response[2] != servo_id:
        print(f"  ✗ ID 불일치: 요청={servo_id}, 응답={response[2]}")
        return False

    error_byte = response[4]
    if error_byte != 0:
        errors = []
        if error_byte & (1 << 0): errors.append("전압에러")
        if error_byte & (1 << 2): errors.append("과열")
        if error_byte & (1 << 5): errors.append("과부하")
        print(f"  △ ID {servo_id} 응답 있으나 에러 플래그: {', '.join(errors)} (0x{error_byte:02X})")
    else:
        print(f"  ✓ ID {servo_id} 정상 응답 (에러 없음)")

    return True


def ping_servo(ser, servo_id: int) -> bool:
    packet = build_ping_packet(servo_id)

    ser.reset_input_buffer()
    ser.write(packet)
    time.sleep(0.01)            # 10ms 대기 (응답 수신 여유)

    response = ser.read(6)      # PING 응답은 6바이트
    if len(response) == 0:
        print(f"  ✗ ID {servo_id}: 응답 없음 (연결 안 됨 또는 ID 없음)")
        return False

    return parse_ping_response(servo_id, response)


def main():
    print("=" * 50)
    print(" Feetech STS3215 PING 테스트")
    print("=" * 50)
    print(f" 포트: {PORT}  |  Baudrate: {BAUDRATE:,}")
    print("=" * 50)

    with serial.Serial(PORT, BAUDRATE, timeout=0.05) as ser:
        found = []

        for servo_id in range(1, 7):    # ID 1~6 스캔
            print(f"\n{'─'*40}")
            ok = ping_servo(ser, servo_id)
            if ok:
                found.append(servo_id)

    print(f"\n{'='*50}")
    print(f" 스캔 결과: {len(found)}개 응답")
    print(f" 응답 ID  : {found if found else '없음'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
