import torch
import pypose as pp
from pypose.module.controllers.geometric_controller import GeometricController
from examples.module.dynamics.multicopter import MultiCopter
from pypose.module.controller_parameters_tuner import ControllerParametersTuner
from examples.module.controller_parameters_tuner.commons \
  import quaternion_2_rotation_matrix, rotation_matrix_2_quaternion


def get_ref_states(waypoints, dt):
    traj_generator = MinimumSnapTrajectoryGenerator()
    quad_trajectory = traj_generator.generate_trajectory(waypoints, dt, 7)

    # get ref states
    ref_states = []
    last_ref_pose = quaternion_2_rotation_matrix(pose)
    last_ref_angle_dot = torch.zeros(3, device=device).double()
    last_ref_angle_ddot = torch.zeros(3, device=device).double()
    ref_states.append(
      (torch.zeros(3, device=device).double(),
      torch.zeros(3, device=device).double(),
      torch.zeros(3, device=device).double(),
      last_ref_pose, last_ref_angle_dot, last_ref_angle_ddot)
    )

    gravity_acc_tensor = torch.stack([
        torch.tensor(0., device=device),
        torch.tensor(0., device=device),
        torch.tensor(g, device=device)]
    ).double()
    for index, waypoint in enumerate(waypoints[1:]):
        position_tensor = torch.stack(
            [waypoint[0],
             waypoint[1],
             waypoint[2]]).double()

        last_position_tensor = torch.stack(
            [waypoints[index][0],
             waypoints[index][1],
             waypoints[index][2]]).double()

        # velocity computation
        # last_position_tensor = ref_states[index][0]
        velocity_tensor = (position_tensor - last_position_tensor) / dt

        # acceleration computation
        last_velocity_tensor = ref_states[index][1]
        raw_acc_tensor = (velocity_tensor - last_velocity_tensor) / dt

        # minus gravity acc if choose upwards as the positive z-axis
        acc_tensor = raw_acc_tensor - gravity_acc_tensor
        acc_tensor_in_column_vector = torch.unsqueeze(acc_tensor, dim=1)

        # assume the yaw angle stays at 0
        b1_yaw_tensor = torch.stack([
            torch.tensor([1.], device=device),
            torch.tensor([0.], device=device),
            torch.tensor([0.], device=device)]).double()

        b3_ref = -acc_tensor_in_column_vector / torch.norm(acc_tensor_in_column_vector)
        b2_ref = torch.cross(b3_ref, b1_yaw_tensor)
        b2_ref = b2_ref / torch.norm(b2_ref)
        b1_ref = torch.cross(b2_ref, b3_ref)
        pose = (torch.concat([b1_ref, b2_ref, b3_ref], dim=1)).double()
        # angle = 2 * torch.acos(q_err[0])
        # angle_dot = angle / dt * axis
        # angle_ddot = ((angle_dot - last_ref_angle_dot) / dt).double()

        # assign zero pose to all waypoints
        angle = 0.0
        angle_dot = torch.zeros([3,], device=device).double()
        angle_ddot = ((angle_dot - last_ref_angle_dot) / dt).double()

        ref_states.append((position_tensor, velocity_tensor,
                           raw_acc_tensor, pose, angle_dot, angle_ddot))

        last_ref_pose = pose
        last_ref_angle_dot = angle_dot
        last_ref_angle_ddot = angle_ddot

    return ref_states


def compute_loss(dynamic_system, controller, controller_parameters,
                 initial_state, ref_states, dt):
    loss = 0
    system_state = torch.clone(initial_state)
    for index, ref_state in enumerate(ref_states):
      ref_position, ref_velocity, ref_acceleration, \
          ref_pose, ref_angular_vel, ref_angular_acc = ref_state
      controller_input = \
          controller.get_control(controller_parameters, system_state, ref_state, None)
      system_new_state = dynamic_system.state_transition(system_state,
                                                         controller_input, dt)

      position, pose, vel, angular_vel = system_new_state[0:3], system_new_state[3:7], \
          system_new_state[7:10], system_new_state[10:13]

      system_state = system_new_state

      loss += torch.norm(
        ref_position - position
      )
    return loss / len(ref_states)


def func_to_get_state_error(state, ref_state):
    ref_position, ref_velocity, ref_acceleration, \
      ref_pose, ref_angular_vel, ref_angular_acc = ref_state

    return state - torch.concat(
       [
          ref_position,
          torch.squeeze(torch.t(rotation_matrix_2_quaternion(ref_pose))),
          ref_velocity,
          ref_angular_vel
       ]
    )


if __name__ == "__main__":
    g = 9.81

    # program parameters
    time_interval = 0.02
    learning_rate = 0.1
    initial_state = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([13, 1]).double()
    initial_controller_parameters = torch.tensor([[0.5003, 0.0996, 0.7173, 0.0100]]).reshape([4, 1]).double()

    points = torch.tensor([[[0., 0., 0.],
                            [1., 1., -2],
                            [3., 3., -4]]], device=args.device)

    waypoints = pp.CSplineR3(points, time_interval)[0]

    ref_states = get_ref_states(quadrotor_waypoints, time_interval)


    e3 = torch.stack([
        torch.tensor([0.]),
        torch.tensor([0.]),
        torch.tensor([1.])]
      ).to(device=args.device).double()
    multicopter = MultiCopter(time_interval,
                               0.6,
                               torch.tensor(g),
                               torch.tensor([
                                  [0.0829, 0., 0.],
                                  [0, 0.0845, 0],
                                  [0, 0, 0.1377]
                                ]))

    # start to tune the controller parameters
    max_tuning_iterations = 30
    tuning_times = 0
    tuner = ControllerParametersTuner(learning_rate=learning_rate)

    controller = SE3Controller(multicopter.m, multicopter.J)
    controller_parameters = torch.clone(initial_controller_parameters)

    # only tune positions
    states_to_tune = torch.zeros([len(initial_state), len(initial_state)])
    states_to_tune[0, 0] = 1
    states_to_tune[1, 1] = 1
    states_to_tune[2, 2] = 1

    print("Original Loss: ", compute_loss(multicopter, controller, controller_parameters,
                                     initial_state, ref_states, time_interval))

    while tuning_times < max_tuning_iterations:
        controller_parameters = tuner.tune(
          multicopter,
          initial_state,
          ref_states,
          controller,
          controller_parameters,
          (0.01 * torch.ones_like(controller_parameters),
            10 * torch.ones_like(controller_parameters)),
          time_interval,
          states_to_tune,
          func_to_get_state_error
        )
        tuning_times += 1
        print("Controller parameters: ", controller_parameters)
        print("Loss: ", compute_loss(multicopter, controller, controller_parameters,
                                     initial_state, ref_states, time_interval))
