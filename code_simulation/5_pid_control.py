import numpy as np
import matplotlib.pyplot as plt

class VirtualMotor:
    def __init__(self, inertia=0.005, damping=0.01):
        self.J = inertia
        self.B = damping
        self.q = 0.0
        self.dq = 0.0
        
    def step(self, torque, dt):
        # 2차 동역학 시스템 모델링 (ddq = (tau - B*dq)/J)
        ddq = (torque - self.B * self.dq) / self.J
        self.dq += ddq * dt
        self.q += self.dq * dt
        return self.q, self.dq

if __name__ == "__main__":
    dt = 0.001
    t_end = 1.0
    time_steps = np.arange(0, t_end, dt)
    
    # 타겟 각도: 45도
    target_q = np.radians(45.0)
    
    # PID 제어 게인 세팅
    Kp, Ki, Kd = 15.0, 5.0, 0.4
    
    motor = VirtualMotor()
    history_q = []
    integral_error = 0.0
    prev_error = 0.0
    
    for t in time_steps:
        curr_q, curr_dq = motor.q, motor.dq
        error = target_q - curr_q
        
        integral_error += error * dt
        derivative_error = (error - prev_error) / dt
        prev_error = error
        
        # PID 제어 입력 연산
        control_torque = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
        
        # 모터 물리 업데이트
        motor.step(control_torque, dt)
        history_q.append(np.degrees(curr_q))
        
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, history_q, label='Response', color='b', linewidth=2)
    plt.axhline(45.0, color='r', linestyle='--', label='Target (45 deg)')
    plt.title("Single-Joint 1-DOF PID Position Control Simulation")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)"); plt.grid(True); plt.legend()
    plt.show()