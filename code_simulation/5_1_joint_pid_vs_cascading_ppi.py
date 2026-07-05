import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


soarm_dh_params = [
    {'alpha': 0.0,       'a': 0.000, 'd': 0.119, 'theta_offset': 0.0},
    {'alpha': -np.pi/2,  'a': 0.068, 'd': 0.000, 'theta_offset': -np.pi/2},
    {'alpha': 0.0,       'a': 0.111, 'd': 0.000, 'theta_offset': 0.0},
    {'alpha': 0.0,       'a': 0.137, 'd': 0.000, 'theta_offset': -np.pi/2},
    {'alpha': -np.pi/2,  'a': 0.000, 'd': 0.099, 'theta_offset': 0.0},
    {'alpha': 0.0,       'a': 0.000, 'd': 0.060, 'theta_offset': 0.0},
]


def dh_transform_matrix(theta, d, a, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct,     -st,     0.0,       a],
        [st*ca,  ct*ca,  -sa,  -sa*d],
        [st*sa,  ct*sa,   ca,   ca*d],
        [0.0,     0.0,   0.0,     1.0],
    ])


def calculate_forward_kinematics(joint_angles):
    q = np.asarray(joint_angles, dtype=float)
    positions = [np.array([0.0, 0.0, 0.0, 1.0])]
    T = np.eye(4)

    for i, params in enumerate(soarm_dh_params):
        joint_theta = q[i] if i < len(q) else 0.0
        theta = joint_theta + params['theta_offset']
        T = T @ dh_transform_matrix(theta, params['d'], params['a'], params['alpha'])
        positions.append(T[:, 3])

    return np.array(positions)[:, :3]


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class VirtualFiveJointArm:
    def __init__(self, q0):
        self.q = np.array(q0, dtype=float)
        self.dq = np.zeros(5)
        self.inertia = np.array([0.018, 0.024, 0.020, 0.010, 0.006])
        self.damping = np.array([0.045, 0.055, 0.050, 0.025, 0.018])
        self.gravity = np.array([0.00, 0.18, 0.13, 0.055, 0.015])

    def step(self, torque, dt):
        torque = np.clip(torque, -1.2, 1.2)
        gravity_load = self.gravity * np.sin(self.q)
        ddq = (torque - self.damping * self.dq - gravity_load) / self.inertia
        self.dq += ddq * dt
        self.q = wrap_to_pi(self.q + self.dq * dt)
        return self.q.copy(), self.dq.copy(), ddq.copy()


class IndependentJointPID:
    def __init__(self):
        self.kp = np.array([12.0, 13.0, 11.0, 7.0, 5.0])
        self.ki = np.array([1.2, 1.4, 1.0, 0.45, 0.25])
        self.kd = np.array([0.55, 0.62, 0.50, 0.22, 0.16])
        self.integral = np.zeros(5)

    def reset(self):
        self.integral[:] = 0.0

    def compute(self, q_des, q, dq, dt):
        error = wrap_to_pi(q_des - q)
        self.integral = np.clip(self.integral + error * dt, -0.6, 0.6)
        torque = self.kp * error + self.ki * self.integral - self.kd * dq
        return np.clip(torque, -1.2, 1.2), error


class CascadingPPIController:
    def __init__(self):
        self.kp_pos = np.array([3.8, 3.4, 3.2, 2.7, 2.2])
        self.kp_vel = np.array([0.34, 0.38, 0.34, 0.20, 0.15])
        self.ki_vel = np.array([1.9, 2.1, 1.8, 0.85, 0.55])
        self.vel_integral = np.zeros(5)
        self.vel_limit = np.deg2rad(np.array([95, 85, 85, 120, 140]))

    def reset(self):
        self.vel_integral[:] = 0.0

    def compute(self, q_des, q, dq, dt):
        pos_error = wrap_to_pi(q_des - q)
        dq_des = np.clip(self.kp_pos * pos_error, -self.vel_limit, self.vel_limit)

        vel_error = dq_des - dq
        self.vel_integral = np.clip(self.vel_integral + vel_error * dt, -1.0, 1.0)
        torque = self.kp_vel * vel_error + self.ki_vel * self.vel_integral
        return np.clip(torque, -1.2, 1.2), pos_error, dq_des


def simulate_controller(controller, controller_type, q0, q_des, duration=3.0, dt=0.002):
    arm = VirtualFiveJointArm(q0)
    controller.reset()

    steps = int(duration / dt) + 1
    t_arr = np.linspace(0.0, duration, steps)
    q_hist = np.zeros((steps, 5))
    dq_hist = np.zeros((steps, 5))
    tau_hist = np.zeros((steps, 5))
    err_hist = np.zeros((steps, 5))
    tcp_hist = np.zeros((steps, 3))
    dq_des_hist = np.zeros((steps, 5))

    for k, _ in enumerate(t_arr):
        q_hist[k] = arm.q
        dq_hist[k] = arm.dq
        tcp_hist[k] = calculate_forward_kinematics(arm.q)[-1]

        if controller_type == 'joint_pid':
            torque, error = controller.compute(q_des, arm.q, arm.dq, dt)
        else:
            torque, error, dq_des = controller.compute(q_des, arm.q, arm.dq, dt)
            dq_des_hist[k] = dq_des

        tau_hist[k] = torque
        err_hist[k] = error
        arm.step(torque, dt)

    return {
        't': t_arr,
        'q': q_hist,
        'dq': dq_hist,
        'tau': tau_hist,
        'err': err_hist,
        'tcp': tcp_hist,
        'dq_des': dq_des_hist,
    }


def settling_time(t_arr, error_norm, threshold=np.deg2rad(2.0)):
    for idx in range(len(t_arr)):
        if np.all(error_norm[idx:] < threshold):
            return t_arr[idx]
    return np.nan


def response_metrics(result, q_des):
    error_norm = np.linalg.norm(result['err'], axis=1)
    final_error_deg = np.rad2deg(error_norm[-1])
    peak_torque = np.max(np.abs(result['tau']))
    settle = settling_time(result['t'], error_norm)
    q_deg = np.rad2deg(result['q'])
    target_deg = np.rad2deg(q_des)
    overshoot = np.max(np.maximum(q_deg - target_deg, 0.0), axis=0)
    undershoot = np.max(np.maximum(target_deg - q_deg, 0.0), axis=0)
    max_overshoot = max(np.max(overshoot), np.max(undershoot))
    return final_error_deg, peak_torque, settle, max_overshoot


def draw_robot(ax, q, color, label, alpha=1.0):
    positions = calculate_forward_kinematics(q)
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        '-o',
        linewidth=3,
        markersize=5,
        color=color,
        alpha=alpha,
        label=label,
    )


def format_summary(title, metrics):
    final_error_deg, peak_torque, settle, max_overshoot = metrics
    settle_text = f"{settle:.2f} s" if not np.isnan(settle) else "not settled"
    return (
        f"{title}\n"
        f"  final error norm: {final_error_deg:6.2f} deg\n"
        f"  peak torque:      {peak_torque:6.2f} Nm\n"
        f"  settling time:    {settle_text}\n"
        f"  max overshoot:    {max_overshoot:6.2f} deg\n"
    )


def plot_comparison(q0, q_des):
    pid = IndependentJointPID()
    ppi = CascadingPPIController()

    pid_result = simulate_controller(pid, 'joint_pid', q0, q_des)
    ppi_result = simulate_controller(ppi, 'cascading_ppi', q0, q_des)

    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title('Joint PID vs Cascading P-PI Control')

    ax_robot = fig.add_subplot(2, 3, 1, projection='3d')
    draw_robot(ax_robot, q0, 'gray', 'Initial', alpha=0.45)
    draw_robot(ax_robot, q_des, 'black', 'Target', alpha=0.55)
    draw_robot(ax_robot, pid_result['q'][-1], 'tab:blue', 'Joint PID final')
    draw_robot(ax_robot, ppi_result['q'][-1], 'tab:orange', 'P-PI final')
    ax_robot.plot(pid_result['tcp'][:, 0], pid_result['tcp'][:, 1], pid_result['tcp'][:, 2],
                  color='tab:blue', linestyle='--', linewidth=1.3, label='PID TCP path')
    ax_robot.plot(ppi_result['tcp'][:, 0], ppi_result['tcp'][:, 1], ppi_result['tcp'][:, 2],
                  color='tab:orange', linestyle='--', linewidth=1.3, label='P-PI TCP path')
    ax_robot.set_title('Robot pose and TCP path')
    ax_robot.set_xlim([-0.45, 0.45])
    ax_robot.set_ylim([-0.45, 0.45])
    ax_robot.set_zlim([0.0, 0.65])
    ax_robot.set_xlabel('X (m)')
    ax_robot.set_ylabel('Y (m)')
    ax_robot.set_zlabel('Z (m)')
    ax_robot.legend(fontsize='small')

    labels = [f'J{i+1}' for i in range(5)]
    target_deg = np.rad2deg(q_des)

    ax_q = fig.add_subplot(2, 3, 2)
    for j in range(5):
        ax_q.plot(pid_result['t'], np.rad2deg(pid_result['q'][:, j]),
                  color=f'C{j}', linestyle='-', label=f'{labels[j]} PID')
        ax_q.plot(ppi_result['t'], np.rad2deg(ppi_result['q'][:, j]),
                  color=f'C{j}', linestyle='--', label=f'{labels[j]} P-PI')
        ax_q.axhline(target_deg[j], color=f'C{j}', alpha=0.18)
    ax_q.set_title('Joint angle response')
    ax_q.set_xlabel('Time (s)')
    ax_q.set_ylabel('Angle (deg)')
    ax_q.grid(True)

    ax_err = fig.add_subplot(2, 3, 3)
    ax_err.plot(pid_result['t'], np.rad2deg(np.linalg.norm(pid_result['err'], axis=1)),
                color='tab:blue', label='Joint PID')
    ax_err.plot(ppi_result['t'], np.rad2deg(np.linalg.norm(ppi_result['err'], axis=1)),
                color='tab:orange', label='P-PI cascading')
    ax_err.set_title('Position error norm')
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Error norm (deg)')
    ax_err.grid(True)
    ax_err.legend()

    ax_tau = fig.add_subplot(2, 3, 4)
    ax_tau.plot(pid_result['t'], np.max(np.abs(pid_result['tau']), axis=1),
                color='tab:blue', label='Joint PID')
    ax_tau.plot(ppi_result['t'], np.max(np.abs(ppi_result['tau']), axis=1),
                color='tab:orange', label='P-PI cascading')
    ax_tau.set_title('Max absolute torque command')
    ax_tau.set_xlabel('Time (s)')
    ax_tau.set_ylabel('Torque (Nm)')
    ax_tau.grid(True)
    ax_tau.legend()

    ax_vel = fig.add_subplot(2, 3, 5)
    ax_vel.plot(ppi_result['t'], np.rad2deg(ppi_result['dq'][:, 1]),
                color='tab:orange', label='J2 actual velocity')
    ax_vel.plot(ppi_result['t'], np.rad2deg(ppi_result['dq_des'][:, 1]),
                color='tab:red', linestyle='--', label='J2 velocity command')
    ax_vel.set_title('P-PI inner velocity loop example')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (deg/s)')
    ax_vel.grid(True)
    ax_vel.legend()

    ax_text = fig.add_subplot(2, 3, 6)
    ax_text.axis('off')
    pid_metrics = response_metrics(pid_result, q_des)
    ppi_metrics = response_metrics(ppi_result, q_des)
    summary = (
        "Independent Joint PID vs Cascading P-PI\n\n"
        "Joint PID:\n"
        "  q error -> torque command directly\n\n"
        "Cascading P-PI:\n"
        "  position P -> velocity command\n"
        "  velocity PI -> torque command\n\n"
        + format_summary('Joint PID result', pid_metrics)
        + "\n"
        + format_summary('P-PI result', ppi_metrics)
    )
    ax_text.text(0.0, 1.0, summary, va='top', family='monospace', fontsize=10)

    handles, labels = ax_q.get_legend_handles_labels()
    ax_q.legend(handles[:10], labels[:10], fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    q0 = np.radians([0.0, -55.0, 70.0, -20.0, 0.0])
    default_target = np.radians([25.0, -25.0, 45.0, 20.0, 35.0])

    setup_fig = plt.figure(figsize=(8, 4.8))
    setup_fig.canvas.manager.set_window_title('Set target joints for PID comparison')
    setup_fig.suptitle('Set target joint angles, then run the comparison')

    sliders = []
    for i in range(5):
        ax_s = setup_fig.add_axes([0.16, 0.76 - i * 0.12, 0.68, 0.04])
        slider = Slider(
            ax_s,
            f'J{i+1} target (deg)',
            -100.0 if i != 4 else -160.0,
            100.0 if i != 4 else 160.0,
            valinit=np.rad2deg(default_target[i]),
        )
        sliders.append(slider)

    ax_run = setup_fig.add_axes([0.22, 0.06, 0.24, 0.08])
    ax_reset = setup_fig.add_axes([0.54, 0.06, 0.24, 0.08])
    btn_run = Button(ax_run, 'Run comparison')
    btn_reset = Button(ax_reset, 'Reset target')

    def current_target():
        return np.radians([s.val for s in sliders])

    def on_run(_event):
        plot_comparison(q0, current_target())

    def on_reset(_event):
        for slider, value in zip(sliders, np.rad2deg(default_target)):
            slider.set_val(value)

    btn_run.on_clicked(on_run)
    btn_reset.on_clicked(on_reset)
    plt.show()
