import torch
from pypose.module.pid import PID

class DubinCarController(PID):
    def __init__(self, parameters):
        super(DubinCarController, self).__init__(parameters)

    def compute_position_error(state, ref_state):
        x_desired, y_desired, v_desired, acc_desired, \
          ori_desired, w_desired, wdot_desired = ref_state
        px, py, orientation, vel, w = state
        ori_cos = torch.cos(orientation)
        ori_sin = torch.sin(orientation)
        return ori_cos * (x_desired - px) + ori_sin * (y_desired - py)

    def

    def forward(self, state, ref_state, dt, feed_forward_quantity):
        x_desired, y_desired, v_desired, acc_desired, \
          ori_desired, w_desired, wdot_desired = ref_state
        px, py, orientation, vel, w = state
        kp, kv, kori, kw = self.parameters

        ori_cos = torch.cos(orientation)
        ori_sin = torch.sin(orientation)
        des_ori_cos = torch.cos(ori_desired)
        des_ori_sin = torch.sin(ori_desired)

        # acceleration output
        acceleration = \
          kp * () \
          + kv * (ori_cos * (v_desired * des_ori_cos - vel * ori_cos)
            + ori_sin * (v_desired * des_ori_sin - vel * ori_sin)) \
          + ori_cos * (acc_desired * des_ori_cos - v_desired * w_desired * des_ori_sin) \
            + ori_sin * (acc_desired * des_ori_sin + v_desired * w_desired * des_ori_cos)

        err_angle = ori_desired - orientation
        orientation_ddot = kori * err_angle + kw * (w_desired - w) + wdot_desired

        return torch.stack([acceleration, orientation_ddot])
