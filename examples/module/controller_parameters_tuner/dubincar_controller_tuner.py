import torch
import argparse, os
import pypose as pp
import matplotlib.pyplot as plt
from pypose.module.dubincar_controller import DubinCarController
from examples.module.dynamics.dubincar import DubinCar
from pypose.module.controller_parameters_tuner import ControllerParametersTuner


def get_ref_states(time, dt, device):
    """
    Generate the trajectory based on constant angular velocity and constant linear velocity
    """
    car_desired_states = []

    waypoint = torch.zeros(2, device=device)
    orientation = torch.zeros(1, device=device)
    angular_vel = torch.ones(1, device=device)
    desired_vel = torch.ones(1, device=device)
    zero_tensor = torch.zeros(1, device=device)

    # get the waypoints and orientation at each timestamp using euler intergration
    for index in range(1, int(time / dt)):
        vel = desired_vel * torch.tensor([torch.cos(orientation), torch.sin(orientation)], device=device)
        waypoint += vel * dt
        orientation += angular_vel * dt

        car_desired_states.append(torch.tensor(
          [waypoint[0], waypoint[1], desired_vel, zero_tensor, \
            orientation, angular_vel, zero_tensor], device=device))
    return car_desired_states

def run(dynamic_system, controller, controller_parameters, initial_state, ref_states, dt):
    system_states = []
    system_state = torch.clone(initial_state)
    system_states.append(system_state)
    for index, ref_state in enumerate(ref_states):
        controller_input = controller.get_control(controller_parameters, system_state, ref_state, None)
        system_new_state = dynamic_system.state_transition(system_state, controller_input, dt)

        system_state = system_new_state
        system_states.append(system_state)
    return system_states

def compute_loss(dynamic_system, controller, controller_parameters, penalty_coefficient,
                 initial_state, ref_states, dt):
    loss = 0
    system_state = torch.clone(initial_state)
    for index, ref_state in enumerate(ref_states):
        ref_position, ref_velocity, ref_acceleration, ref_pose, ref_angular_vel, ref_angular_acc = \
          ref_state[0:2], ref_state[2:4], ref_state[4:6], \
          ref_state[6:7], ref_state[7:8], ref_state[8:9]
        controller_input = controller.get_control(controller_parameters, system_state, ref_state, None)
        system_new_state = dynamic_system.state_transition(system_state, controller_input, dt)

        position_x, position_y, pose, vel, angular_vel = system_new_state

        system_state = system_new_state

        loss += torch.norm(
          ref_position - torch.stack([position_x, position_y])
        )
        loss += penalty_coefficient * torch.norm(controller_input)
    return loss / len(ref_states)

def func_to_get_state_error(state, ref_state):
    x_desired, y_desired, v_desired, acc_desired, \
      ori_desired, w_desired, wdot_desired = ref_state

    return state - torch.stack(
       [
          x_desired,
          y_desired,
          ori_desired,
          v_desired,
          w_desired
       ])

def get_sub_states(input_states, sub_state_index):
    sub_states = []
    for states in input_states:
        sub_states.append(states[sub_state_index].detach().cpu().item())
    return sub_states

def subPlot(ax, x, y, y_ref, xlabel=None, ylabel=None):
    res = []
    for i in range(0, len(y)):
        res.append(y_ref[i] - y[i])
    ax.plot(x, res, label=ylabel)
    ax.legend()
    ax.set_xlabel(xlabel)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dubincar Controller Tuner Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/controller_parameters_tuner/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    # program parameters
    time_interval = 0.02
    learning_rate = 10
    # states tensor: x position, y position, orientation, velocity, angular_velocity
    initial_state = torch.zeros(5, device=args.device).double()
    # controller parameters: kp_position, kp_velocity, kp_orientation, kp_angular_velocity
    initial_controller_parameters = torch.ones(4, device=args.device).double() * 5

    ref_states = get_ref_states(10, time_interval, args.device)

    dubincar = DubinCar()

    # start to tune the controller parameters
    penalty_coefficient = 0.0
    tuner = ControllerParametersTuner(learning_rate=learning_rate,
                                      penalty_coefficient=penalty_coefficient,
                                      device=args.device)

    controller = DubinCarController()
    controller_parameters = torch.clone(initial_controller_parameters)

    states_to_tune = torch.zeros([len(initial_state), len(initial_state)]
      , device=args.device)
    # only to tune the controller parameters dependending on the position error
    states_to_tune[0, 0] = 1
    states_to_tune[1, 1] = 1

    last_loss_after_tuning = compute_loss(dubincar, controller, controller_parameters,
                                          penalty_coefficient, initial_state, ref_states, time_interval)
    print("Original Loss: ", last_loss_after_tuning)

    meet_termination_condition = False
    while not meet_termination_condition:
        controller_parameters = tuner.tune(
          dubincar,
          initial_state,
          ref_states,
          controller,
          controller_parameters,
          (0.001 * torch.ones_like(controller_parameters),
            20 * torch.ones_like(controller_parameters)),
          time_interval,
          states_to_tune,
          func_to_get_state_error
        )
        print("Controller parameters: ", controller_parameters)

        loss = compute_loss(dubincar, controller, controller_parameters, penalty_coefficient,
                            initial_state, ref_states, time_interval)
        print("Loss: ", loss)

        if (last_loss_after_tuning - loss) < 0.00001:
            meet_termination_condition = True
            print("Meet tuning termination condition, terminated.")
        else:
            last_loss_after_tuning = loss

    # plot the result
    # get the result with original and tuned controller parameters
    original_system_states = run(dubincar, controller, torch.clone(initial_controller_parameters),
                                 initial_state, ref_states, time_interval)
    new_system_states = run(dubincar, controller, controller_parameters, initial_state,
                                  ref_states, time_interval)
    ref_states.insert(0, torch.zeros(9, device=args.device))
    time = torch.arange(0, time_interval * len(ref_states), time_interval).numpy()
    f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].set_title("Tracking result before tuning")
    ax[0].set_xlabel("time(s)")
    ax[0].set_ylabel("Tracking error(m)")
    ax[1].set_title("Tracking result after tuning")
    ax[1].set_xlabel("time(s)")
    ax[1].set_ylabel("Tracking error(m)")
    subPlot(ax[0], time, get_sub_states(original_system_states, 0), get_sub_states(ref_states, 0), ylabel='X')
    subPlot(ax[0], time, get_sub_states(original_system_states, 1), get_sub_states(ref_states, 1), ylabel='Y')
    subPlot(ax[1], time, get_sub_states(new_system_states, 0), get_sub_states(ref_states, 0), ylabel='X')
    subPlot(ax[1], time, get_sub_states(new_system_states, 1), get_sub_states(ref_states, 1), ylabel='Y')
    figure = os.path.join(args.save + 'dubincar_controller_tuner.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
