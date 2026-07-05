import numpy as np
import matplotlib.pyplot as plt

def profile_cubic(q0, q1, tf, t):
    tau = t / tf
    s = 3*tau**2 - 2*tau**3
    ds = (6*tau - 6*tau**2) / tf
    dds = (6 - 12*tau) / tf**2
    return q0 + (q1-q0)*s, (q1-q0)*ds, (q1-q0)*dds

def profile_quintic(q0, q1, tf, t):
    tau = t / tf
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    ds = (30*tau**2 - 60*tau**3 + 30*tau**4) / tf
    dds = (60*tau - 180*tau**2 + 120*tau**3) / tf**2
    return q0 + (q1-q0)*s, (q1-q0)*ds, (q1-q0)*dds

def profile_trapezoidal(q0, q1, tf, t):
    ta = tf / 3.0
    v_max = 1.0 / (tf - ta)
    a_max = v_max / ta
    dist = q1 - q0
    if t <= ta:
        s = 0.5 * a_max * t**2
        ds = a_max * t
        dds = a_max
    elif t <= tf - ta:
        s = 0.5 * a_max * ta**2 + v_max * (t - ta)
        ds = v_max
        dds = 0.0
    else:
        td = t - (tf - ta)
        s = 0.5 * a_max * ta**2 + v_max * (tf - 2*ta) + v_max * td - 0.5 * a_max * td**2
        ds = v_max - a_max * td
        dds = -a_max
    return q0 + dist * s, dist * ds, dist * dds

if __name__ == "__main__":
    t_arr = np.linspace(0, 2.0, 200)
    q0, q1, tf = 0.0, 90.0, 2.0
    
    res = {'c': [], 'q': [], 't': []}
    for t in t_arr:
        res['c'].append(profile_cubic(q0, q1, tf, t))
        res['q'].append(profile_quintic(q0, q1, tf, t))
        res['t'].append(profile_trapezoidal(q0, q1, tf, t))
        
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    profiles = [('Cubic (3rd)', 'c', 'blue'), ('Quintic (5th)', 'q', 'green'), ('Trapezoidal', 't', 'red')]
    
    for label, key, color in profiles:
        data = np.array(res[key])
        axs[0].plot(t_arr, data[:, 0], label=label, color=color)
        axs[1].plot(t_arr, data[:, 1], color=color)
        axs[2].plot(t_arr, data[:, 2], color=color)
        
    axs[0].set_ylabel('Position (deg)'); axs[0].legend(); axs[0].grid(True)
    axs[1].set_ylabel('Velocity (deg/s)'); axs[1].grid(True)
    axs[2].set_ylabel('Acceleration (deg/s^2)'); axs[2].grid(True)
    axs[2].set_xlabel('Time (seconds)')
    
    axs[0].set_title("Trajectory Profile Comparison (Notice the Jerk at boundaries!)")
    plt.tight_layout()
    plt.show()