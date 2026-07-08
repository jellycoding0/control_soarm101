import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 링크의 길이
a1 = 1.0  # 첫 번째 링크 길이
a2 = 1.0  # 두 번째 링크 길이

def forward_kinematics(theta1, theta2):
    """엔드 이펙터의 위치 계산 (정기구학)"""
    x1 = a1 * np.cos(theta1)
    y1 = a1 * np.sin(theta1)
    x2 = x1 + a2 * np.cos(theta1 + theta2)
    y2 = y1 + a2 * np.sin(theta1 + theta2)
    return np.array([x1, y1]), np.array([x2, y2])

def jacobian(theta1, theta2):
    """자코비안 계산 (이론식과 매칭되도록 부호 교정)"""
    j11 = -a1 * np.sin(theta1) - a2 * np.sin(theta1 + theta2)
    j12 = -a2 * np.sin(theta1 + theta2)
    j21 = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
    j22 = a2 * np.cos(theta1 + theta2)
    return np.array([[j11, j12], [j21, j22]])

def check_singularity(J):
    """특이점 확인"""
    det_J = np.linalg.det(J)
    return abs(det_J) < 1e-6, det_J

# 1행 2열(좌우 배치) 구조로 변경, 가로가 더 긴 figsize=(12, 6)로 설정
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# 하단 슬라이더 배치를 위한 아래쪽 여백(bottom=0.2) 및 좌우 간격(wspace) 확보
plt.subplots_adjust(bottom=0.2, wspace=0.3)

ax1.set_xlim(-2.2, 2.2)
ax1.set_ylim(-2.2, 2.2)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title("Robot Arm Simulation")

# 가로 배치에 맞게 하단 슬라이더 위치와 너비를 널찍하게 재조정
# [좌측 시작점, 아래 시작점, 가로 너비, 세로 높이]
slider_ax1 = plt.axes([0.2, 0.08, 0.6, 0.025])
slider_ax2 = plt.axes([0.2, 0.03, 0.6, 0.025])

# 슬라이더 초기화
theta1_slider = Slider(slider_ax1, 'Theta 1 (rad)', -np.pi, np.pi, valinit=0.0)
theta2_slider = Slider(slider_ax2, 'Theta 2 (rad)', -np.pi, np.pi, valinit=0.0)

def draw_robot(theta1, theta2):
    """로봇 팔 및 자코비안 인디케이터 실시간 갱신"""
    ax1.clear()
    ax1.set_xlim(-2.2, 2.2)
    ax1.set_ylim(-2.2, 2.2)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title("Robot Arm Simulation")
    
    joint1 = np.array([0, 0])
    joint2, end_effector = forward_kinematics(theta1, theta2)

    # 링크 및 조인트 그리기
    ax1.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 'ro-', lw=3, label='Link 1')
    ax1.plot([joint2[0], end_effector[0]], [joint2[1], end_effector[1]], 'bo-', lw=3, label='Link 2')
    ax1.plot(end_effector[0], end_effector[1], 'go', ms=10, label='End Effector')
    ax1.legend(loc='upper right')

    # 자코비안 및 특이점 계산
    J = jacobian(theta1, theta2)
    singular, det_J = check_singularity(J)
    
    # 행렬식 값 시각화 업데이트
    ax2.clear()
    ax2.bar(['det(J)'], [det_J], color='red' if singular else 'green', width=0.4)
    ax2.set_title("Singularity Determinant Indicator")
    ax2.set_ylabel("det_Jacobian")
    ax2.set_ylim(-1.2, 1.2)
    ax2.axhline(0, color='black', lw=1.5)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5) # 바 차트 가독성을 위한 y축 그리드 추가

    # 특이점 여부 경고 출력
    if singular:
        ax2.text(0, 0.2, '⚠️ SINGULARITY!', color='red', fontsize=12, fontweight='bold', ha='center')

def update(val):
    theta1 = theta1_slider.val
    theta2 = theta2_slider.val
    draw_robot(theta1, theta2)

# 슬라이더 이벤트 바인딩
theta1_slider.on_changed(update)
theta2_slider.on_changed(update)

# 초기 화면 출력
draw_robot(0.0, 0.0)
plt.show()