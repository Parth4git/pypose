import torch
from torch import nn
from torch.autograd.functional import jacobian


class ControllerParametersTuner(nn.Module):
    r"""
    Args:
            dynamic_system (pypose.module.dynamics): dynamics system
            controller (pypose.module.controller): Linear or nonlinear controller to control the dynamic system
            parameters_bound (Tensor): This set gives the minimum and the maximum
                value the parameters can reach
            states_to_tune (Tensor): choose which state needs to be considered in the
                loss function, usually only position is chosen.
            func_get_state_error (function): function has two inputs: system state and
                ref state. The function needs to be provided by users considering system
                state and ref state are not always in the same formation or dimension.
            learning_rate(float): gradient descent step size
            penalty_coefficient(float): controller-effort penalty coefficient
            device(string): on the cpu or cuda to perform the tuning process


    This class is the general implementation of the controller parameters tuner based on
    the sensitivity propagation.

    Given a dynamic system, the system state transition can be defined as

    .. math::
            \mathbf{x_{i+1}} = f(\mathbf{x_i}, \mathbf{u_i}) \tag{1}

    .. math::
            \mathbf{u_i} = h(\mathbf{x_i}, \mathbf{\hat{x_i}}, \pmb{\pmb{\theta}}) \tag{2}

    Where :math:`\mathbf{x_{i+1}}` stands for the dynmamic system states at time :math:`i+1`,
    :math:`\mathbf{x_i} \in \mathbb{R}^n` gives the dynamic system states at time :math:`i`,
    :math:`\mathbf{\hat x_i} \in \mathbb{R}^n` gives the system reference states at time :math:`i`,
    :math:`\mathbf{u_i} \in \mathbb{R}^m` is the controller-generated system inputs at time :math:`i`,
    :math:`\mathbf{\pmb{\theta}}` is the controller parameters.

    Note:
        For the parameter initial_state and parameters, please use 1d row vector to
        represent these variables at the same time.



    Given a dynamic system, a corresponding controller and reference states, assuming all
    these components are differentiable, it is possbile to find the most suitable
    parameters by constructing the loss function :math:`L` and computing the gradient of
    :math:`L` with respect to controller parameters.



    .. math::
            L = \sum_{i=1}^N ||\mathbf{x_i} - \mathbf{\hat x_i}||^2 + \sum_{i=0}^{N-1} \lambda|| \mathbf{u_i} ||^2 \tag{3}

    Where :math:`L` is the Loss function. Here it works as the control-effort penalty in the loss function. :math:`\lambda` is the
    penalty coefficient.

    And we can get the derivatives as following:

    .. math::
        \begin{align*}
            \nabla L_{\pmb{\theta}} &= \sum_{i=1}^N \frac{\partial L}{\partial \mathbf{x_i}} \frac{\partial \mathbf{x_i}}{\partial \pmb{\theta}} + \sum_{i=0}^{N-1} \frac{\partial L}{\partial \mathbf{u_i}} \frac{\partial \mathbf{u_i}}{\partial \pmb{\theta}} \\
            \frac{\partial \mathbf{x_{i+1}}}{\partial \pmb{\theta}} &= (\frac{\partial \mathbf{x_{i+1}}}{\partial \mathbf{x_i}} + \frac{\partial \mathbf{x_{i+1}}}{\partial \mathbf{u_i}}\frac{\partial \mathbf{u_i}}{\partial \mathbf{x_i}})
            \frac{\partial \mathbf{x_i}}{\partial \pmb{\theta}}+\frac{\partial \mathbf{x_{i+1}}}{\partial \mathbf{u_i}}\frac{\partial \mathbf{u_i}}{\partial \pmb{\theta}} \\
            \frac{\partial \mathbf{u_i}}{\partial \pmb{\theta}} &= \frac{\partial \mathbf{u_i}}{\partial \mathbf{x_i}} \frac{\partial \mathbf{x_i}}{\partial \pmb{\theta}} + \frac{\partial \mathbf{u_i}}{\partial \pmb{\theta}}
        \end{align*} \tag{4}

    To make sure the controller parameters can stay in the reasonable and safe sets, after updating the parameters, it's essential to project the new parameters into
    the feasiable set. And the parameters can be updated using the following equation:

    .. math::
        \pmb{\theta} \leftarrow P_{\Theta}(\pmb{\theta} - \alpha\nabla_{\pmb{\theta}}L) \tag{5}


    The whole controller parameters tuning pipeline can be found here:

    .. math::
        \begin{aligned}
            &\rule{120mm}{0.4pt}                                                                        \\
            &\textbf{input}: \text{Initial state }\mathbf{\bar x_0}, \text{initial controller parameters }\pmb{\theta}_0,
            \text{feasible set }\Theta, \text{horizon }N,                                               \\
            &\hspace{12mm} \text{desired states }\mathbf{\hat{x}_{1:N}}, \text{step size }\alpha,
            \text{and termination condiction } C                                                        \\

            &\rule{120mm}{0.4pt}                                                                        \\
            &\textbf{Output} \: \text{Tuned parameter: } \pmb{\theta}^{\ast}                                  \\
            &\hspace{5mm} \text{Initialize }\pmb{\theta} \leftarrow \pmb{\theta}_0                                  \\
            &\hspace{5mm} \text{While }C \text{ is FALSE }\textbf{do}                                   \\
            &\hspace{10mm} \text{Reset } \mathbf{x_0} \text{ to } \mathbf{\bar x_0}.                                       \\
            &\hspace{10mm} \textbf{for } k \leftarrow 0 \text{ to } N \textbf{ do}                           \\
            &\hspace{15mm} \text{Obtain } \mathbf{x_i} \text{ from system and compute } \mathbf{x_{i+1}} \text{ and } \mathbf{u_i} \text{ using equation (1) and (2)}  \\
            &\hspace{15mm} \text{Update} \frac{\partial \mathbf{x_i}}{\partial \pmb{\theta}} \text{ and } \frac{\partial \mathbf{u_i}}{\partial \pmb{\theta}} \text{using equation (4)} \\
            &\hspace{15mm} \text{Compute } \frac{\partial L}{\partial \mathbf{x_i}} \text{ and } \frac{\partial L}{\partial \mathbf{u_i}}. \\
            &\hspace{15mm} \text{Store } \mathbf{x_i}, \mathbf{u_i}, \frac{\partial \mathbf{x_{i+1}}}{\partial \pmb{\theta}}, \frac{\partial \mathbf{u_i}}{\partial \pmb{\theta}}, \frac{\partial L}{\partial \mathbf{x_i}}\text{ and } \frac{\partial L}{\partial \mathbf{u_i}}\text{ in memory}. \\
            &\hspace{10mm} \textbf{end for} \\
            &\hspace{10mm} \text{Compute }\nabla L_{\pmb{\theta}} \text{ using equation (4) and update } \pmb{\theta} \text{with equation (5).} \\
            &\hspace{5mm} \textbf{end while}                                                \\
            &\rule{120mm}{0.4pt}                                                            \\[-1.ex]
            &\bf{return} \:  \text{the tuned parameters }\pmb{\theta}^{\ast} \leftarrow \pmb{\theta}                                                    \\[-1.ex]
            &\rule{120mm}{0.4pt}                                                            \\[-1.ex]
       \end{aligned}


    Note:
        This controller parameter tuning algorithm is developed based on the method proposed by this paper:

        * Cheng, Sheng, et al. `DiffTune: Auto-Tuning through Auto-Differentiation. <https://arxiv.org/abs/2209.10021>`_ arXiv preprint arXiv:2209.10021 (2022).


    """
    def __init__(self, dynamic_system, controller, parameters_bound, states_to_tune, \
                 func_get_state_error, learning_rate, penalty_coefficient, device,):
        super().__init__()
        self.learning_rate = learning_rate
        self.device = device
        self.penalty_coefficient = penalty_coefficient
        self.dynamic_system = dynamic_system
        self.controller = controller
        self.parameters_bound = parameters_bound
        self.states_to_tune = states_to_tune
        self.func_get_state_error = func_get_state_error

    def forward(self, initial_state, ref_states, tau):
        r"""
        Args:

            initial_state (Tensor): 1d tensor representing the states of the dynamic system
            ref_states (object): these reference states are defined by the user and no need to be specific formation,
                but it has to be able to be used by the controller anf function func_get_state_error. In this function, we assume the first reference state
                should not be equal to the system initial state.

            tau: time interval considered in system


        Return:
            list of :obj:`Tensor`: tuned controller parameters, original state loss and
            new state loss using tuned controller parameters
        """
        states_to_tune = self.states_to_tune
        states = []
        inputs = []
        dxdparam_gradients = []
        dukdparam_gradients = []

        system_state = torch.clone(initial_state)
        controller_parameters = self.controller.parameters.detach()
        states.append(system_state)
        dxdparam_gradients.append(
            torch.zeros(
                [len(initial_state), len(controller_parameters)], device=self.device))

        for index, ref_state in enumerate(ref_states):
            controller_input = self.controller.forward(state=system_state, \
                                ref_state=ref_state, feed_forward_quantity=None)

            system_new_state = self.dynamic_system.state_transition(system_state, controller_input, tau)

            # calcuate the state derivative wrt. the parameters and the input derivative wrt. the parameters
            dhdx_func = lambda state: self.controller.forward(state = state, \
                                        ref_state = ref_state, feed_forward_quantity = None)
            dhdxk_tensor = torch.squeeze(jacobian(dhdx_func, system_state))

            dhdparam_tensor = torch.zeros(len(controller_input), len(self.controller.parameters))
            for i in range(0, len(controller_input)):
                dhdparam_tensor[i] = torch.autograd.grad(controller_input[i], \
                        self.controller.parameters, retain_graph=True)[0]
            system_state = system_state.detach()

            dfdxk_func = lambda system_state: self.dynamic_system.state_transition(state = system_state,
                                                            input = controller_input, t = tau)
            dfdxk_tensor = torch.squeeze(jacobian(dfdxk_func, system_state))

            dfduk_func = lambda inputs: self.dynamic_system.state_transition(state = system_state,
                                                                        input = inputs, t = tau)
            dfduk_tensor = torch.squeeze(jacobian(dfduk_func, controller_input))

            states.append(system_new_state)
            inputs.append(controller_input)
            system_state = system_new_state.detach()

            last_gradient = dxdparam_gradients[-1]

            dxdparam_gradients.append(
                torch.mm(dfdxk_tensor + torch.mm(dfduk_tensor, dhdxk_tensor), last_gradient) \
                + torch.mm(dfduk_tensor, dhdparam_tensor)
            )

            dukdparam_gradients.append(
                torch.mm(dhdxk_tensor, last_gradient) + dhdparam_tensor
            )

        # accumulate the gradients
        gradient_sum = torch.zeros([len(controller_parameters), 1], device=self.device)
        # error summation between system state and reference state
        loss = torch.zeros(1, device=self.device)

        for ref_state_index in range(0, len(ref_states)):
            state_error = self.func_get_state_error(states[ref_state_index + 1], \
                                               ref_states[ref_state_index])
            state_error = torch.atleast_2d(state_error)
            state_error = torch.mm(state_error, states_to_tune)
            gradient_sum += torch.t(2 * torch.mm(state_error, \
                                                 dxdparam_gradients[ref_state_index]))
            gradient_sum += self.penalty_coefficient * \
                torch.t(2 * torch.mm(torch.atleast_2d(inputs[ref_state_index]), \
                                       dukdparam_gradients[ref_state_index]))
            loss += torch.norm(state_error)

        gradient_sum = torch.squeeze(torch.t(gradient_sum))
        loss /= len(ref_states)

        min_parameters = self.parameters_bound[0]
        max_parameters = self.parameters_bound[1]
        controller_parameters = torch.min(max_parameters, \
            torch.max(min_parameters, controller_parameters - self.learning_rate * gradient_sum))

        # compute the loss using new controller parameters
        self.controller.parameters = controller_parameters.detach()
        self.controller.parameters.requires_grad = True
        new_loss = self.loss_computation(self.dynamic_system, initial_state, ref_states, \
                    self.controller, tau, states_to_tune, self.func_get_state_error)

        return controller_parameters, loss, new_loss

    def loss_computation(self, dynamic_system, initial_state, ref_states, controller,
                         tau, states_to_tune, func_get_state_error):
        """
        Compute the loss besed on the given controller parameters and state error defined
        by the func_get_state_error input parameter.
        """
        system_state = torch.clone(initial_state)
        loss = torch.zeros(1, device=self.device)
        for index, ref_state in enumerate(ref_states):
            controller_input = controller.forward(state=system_state, \
                                    ref_state=ref_state, feed_forward_quantity=None)
            system_new_state = dynamic_system.state_transition(system_state, \
                                                               controller_input, tau)
            state_error = func_get_state_error(system_new_state, ref_state)
            state_error = torch.atleast_2d(state_error)
            state_error = torch.mm(state_error, states_to_tune)
            loss += torch.norm(state_error)

            system_state = system_new_state
        loss /= len(ref_states)
        return loss
