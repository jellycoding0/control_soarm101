import numpy as np


class TrajectoryGenerator:
    @staticmethod
    def get_profile_s(t, T, profile_type='COSINE'):
        if t <= 0: return 0.0
        if t >= T: return 1.0
        if profile_type == 'LINEAR':
            return t / T
        elif profile_type == 'COSINE':
            return (1.0 - np.cos(np.pi * t / T)) / 2.0
        return t / T

    def joint_trajectory(self, q0, q1, duration, dt=0.02, profile='COSINE'):
        steps = int(duration / dt)
        traj = []
        for i in range(steps + 1):
            t = i * dt
            s = self.get_profile_s(t, duration, profile)
            traj.append(q0 + s * (q1 - q0))
        return traj
