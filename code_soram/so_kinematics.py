import numpy as np
from so_servo_bus import load_kinematic_zero_offsets_deg


class Kinematics:
    def __init__(self):
        # Modified DH 파라미터 테이블 (from image 21106.jpg)
        self.dh_params = [
            # [alpha,    a,   d,  theta_offset]
            [0,         0,  119,    0.0],         # Joint 1 (Base)
            [-np.pi/2,  68.0, 0,   -np.pi/2],     # Joint 2 (Shoulder)
            [0,       111,    0,    0.0],         # Joint 3 (Elbow)
            [0,       137,    0,   -np.pi/2],     # Joint 4 (Wrist Pitch) - offset to lift by 90 deg
            [-np.pi/2,  0,   99,    0.0],         # Joint 5 (Wrist Roll)
            [0,         0,   60,    0.0]          # EE (고정 링크 역할)
        ]
        self.kinematic_zero_offsets = np.deg2rad(load_kinematic_zero_offsets_deg())

    def model_to_dh_angles(self, joint_angles):
        q = np.array(joint_angles, dtype=float)
        q_dh = q - self.kinematic_zero_offsets
        return np.arctan2(np.sin(q_dh), np.cos(q_dh))
        
    def get_transform_matrix(self, alpha, a, d, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct,     -st,     0,       a],
            [st*ca,  ct*ca,  -sa,  -sa*d],
            [st*sa,  ct*sa,   ca,   ca*d],
            [0,       0,      0,       1]
        ])

    def forward_kinematics(self, joint_angles):
        thetas = list(self.model_to_dh_angles(joint_angles)) + [0.0]
        T_total = np.eye(4)
        frames = [T_total.copy()]
        
        for i in range(6):
            alpha, a, d, theta_offset = self.dh_params[i]
            T_i = self.get_transform_matrix(alpha, a, d, thetas[i] + theta_offset)
            T_total = T_total @ T_i
            frames.append(T_total.copy())
            
        return T_total, frames

    def get_geometric_jacobian(self, frames):
        J = np.zeros((6, 5))
        p_e = frames[-1][:3, 3]
        
        for i in range(5):
            z = frames[i + 1][:3, 2]
            p = frames[i + 1][:3, 3]
            J[:3, i] = np.cross(z, p_e - p)
            J[3:, i] = z
        return J

    def inverse_kinematics_dls(self, T_target, q_seed):
        lam = 5.0
        alpha_step = 0.5
        tol = 0.0005
        max_iter = 300
        
        q = np.array(q_seed, dtype=float)
        for _ in range(max_iter):
            T_curr, frames = self.forward_kinematics(q)
            p_err = T_target[:3, 3] - T_curr[:3, 3]
            
            R_curr = T_curr[:3, :3]
            R_target = T_target[:3, :3]
            R_err = R_target @ R_curr.T
            
            tr = np.trace(R_err)
            if abs(tr - 3.0) < 1e-6:
                w_err = np.zeros(3)
            else:
                theta_err = np.arccos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))
                if abs(theta_err) < 1e-6:
                    w_err = np.zeros(3)
                else:
                    axis = np.array([
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1]
                    ]) / (2.0 * np.sin(theta_err))
                    w_err = axis * theta_err
                    
            e = np.hstack((p_err, w_err))
            if np.linalg.norm(p_err) < tol:
                break
                
            J = self.get_geometric_jacobian(frames)
            JJT = J @ J.T
            damping = (lam ** 2) * np.eye(6)
            dq = J.T @ np.linalg.inv(JJT + damping) @ e
            q += alpha_step * dq
            
        return q

    def inverse_kinematics_position_dls(self, xyz_target, q_seed, fixed_joint_indices=(4,)):
        lam = 8.0
        alpha_step = 0.6
        tol_mm = 0.2
        max_iter = 200
        max_step_rad = np.deg2rad(2.0)

        q_seed = np.array(q_seed, dtype=float)
        q = q_seed.copy()
        xyz_target = np.array(xyz_target, dtype=float)

        for _ in range(max_iter):
            T_curr, frames = self.forward_kinematics(q)
            p_err = xyz_target - T_curr[:3, 3]
            if np.linalg.norm(p_err) < tol_mm:
                break

            J = self.get_geometric_jacobian(frames)[:3, :]
            for idx in fixed_joint_indices:
                J[:, idx] = 0.0

            JJT = J @ J.T
            dq = J.T @ np.linalg.inv(JJT + (lam ** 2) * np.eye(3)) @ p_err
            dq = np.clip(alpha_step * dq, -max_step_rad, max_step_rad)
            for idx in fixed_joint_indices:
                dq[idx] = 0.0

            q += dq
            q = q_seed + np.arctan2(np.sin(q - q_seed), np.cos(q - q_seed))

        return q
