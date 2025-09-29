import numpy as np
import casadi as ca


# 更改声明的位置
class Control:
    def __init__(self):
        self.N = 5
        self.curr_state = None
        self.ob = []
        self.goal_state = np.array([0.0, 4.0])
        self.goal_state = self.goal_state[np.newaxis, :]

    def curr_pose_cb(self, robot_x, robot_y):
        self.curr_state = np.zeros(2)
        self.curr_state[0] = robot_x
        self.curr_state[1] = robot_y

    def obs_cb(self, pred_list):
        self.ob = []
        size = int(len(pred_list))
        for i in range(size):
            self.ob.append(np.array([pred_list[i].x, pred_list[i].y]))

    def mpc(self, action):
        opti = ca.Opti()
        T = 0.25
        v_max = 1.0
        gamma_k = 0.3
        opt_x0 = opti.parameter(2)
        opt_states = opti.variable(self.N + 1, 2)
        opt_controls = opti.variable(self.N, 2)
        vx = opt_controls[:, 0]
        vy = opt_controls[:, 1]

        # 运动学公式
        def f(u_):
            return ca.vertcat(*[u_[0], u_[1]])

        def h(state, ob_):
            r = 0.3
            ob_vec = ca.MX([ob_[0], ob_[1]])
            center_vec = state[:2] - ob_vec.T
            dist = ca.sqrt(center_vec[0] ** 2 + center_vec[1] ** 2) - 2 * r
            return dist

        def quadratic(x, a):
            return ca.mtimes([x, a, x.T])

        opti.subject_to(opt_states[0, :] == opt_x0.T)

        opti.subject_to(opti.bounded(-v_max, vx, v_max))

        opti.subject_to(opti.bounded(-v_max, vy, v_max))

        for i in range(self.N):
            x_next = opt_states[i, :] + T * f(opt_controls[i, :]).T
            opti.subject_to(opt_states[i + 1, :] == x_next)

        num_obs = int(len(self.ob) / (self.N + 1))

        # CBF的约束方式
        for j in range(num_obs):
            for i in range(self.N):
                opti.subject_to(h(opt_states[i + 1, :], self.ob[(self.N + 1) * j + i + 1]) >=
                                (1 - gamma_k) * h(opt_states[i, :], self.ob[j * (self.N + 1) + i]))

        obj = 0
        R = np.diag([1.0, 1.0])
        action1 = np.zeros(2)
        action1[0] = action.vx
        action1[1] = action.vy
        action2 = action1[np.newaxis, :]
        obj += quadratic(opt_controls[0, :] - action2, R)

        opti.minimize(obj)
        opts_setting = {'ipopt.max_iter': 2000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-3,
                        'ipopt.acceptable_obj_change_tol': 1e-3}

        opti.solver('ipopt', opts_setting)
        opti.set_value(opt_x0, self.curr_state)

        try:
            sol = opti.solve()
            u_res = sol.value(opt_controls)
            return u_res[0]

        except:
            return action1
