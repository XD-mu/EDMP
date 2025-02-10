from torch.optim.sgd import SGD
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import math
import torch

class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b
            self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))

# class MetaAdamW(AdamW):
#     def __init__(self, net, *args, **kwargs):
#         super(MetaAdamW, self).__init__(*args, **kwargs)
#         self.net = net
#
#     def set_parameter(self, current_module, name, parameters):
#         if '.' in name:
#             name_split = name.split('.')
#             module_name = name_split[0]
#             rest_name = '.'.join(name_split[1:])
#             # print("更换了参数")
#             for children_name, children in current_module.named_children():
#                 if module_name == children_name:
#                     self.set_parameter(children, rest_name, parameters)
#
#                     break
#         else:
#             current_module._parameters[name] = parameters
#
#     def meta_step(self, grads):
#         assert len(list(self.net.parameters())) == len(grads), "Parameters and grads must match in length"
#
#         torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)  # 梯度裁剪
#
#         for i, group in enumerate(self.param_groups):
#             group = self.param_groups[i]
#             lr = group['lr']
#             weight_decay = group['weight_decay']
#             for (name, parameter), grad in zip(self.net.named_parameters(), grads):
#                 if grad is not None:
#                     parameter.detach_()
#                     if weight_decay != 0:
#                         # grad_wd = grad.add(parameter.data, alpha=group['weight_decay'])
#                         grad_wd = grad.add(parameter.data, alpha=group['weight_decay'])
#
#                     else:
#                         grad_wd = grad
#                     # grad_wd = grad.add(parameter.data, alpha=group['weight_decay'])
#                     step_size = group['lr']
#                     beta1, beta2 = group['betas']
#
#                     # 更新状态或初始化状态
#                     exp_avg = self.state[parameter].get("exp_avg", torch.zeros_like(parameter.data))
#                     exp_avg_sq = self.state[parameter].get("exp_avg_sq", torch.zeros_like(parameter.data))
#                     step = self.state[parameter].get("step", 0) + 1
#
#                     exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                     exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
#
#                     exp_avg_corr = exp_avg / (1 - beta1 ** step)
#                     exp_avg_sq_corr = exp_avg_sq / (1 - beta2 ** step)
#
#                     denom = (exp_avg_sq_corr.sqrt().add(group['eps']))
#                     step_direction = exp_avg_corr / denom
#                     updated_param = parameter.data - step_size * step_direction
#
#                     self.state[parameter].update({
#                         "exp_avg": exp_avg,
#                         "exp_avg_sq": exp_avg_sq,
#                         "step": step
#                     })
#
#                     # 更新参数的梯度
#                     updated_params = updated_param.data.add(grad_wd, alpha=-lr)
#                     # updated_params = updated_param.add(grad_wd, alpha=-lr)
#                     self.set_parameter(self.net, name, updated_params)
#                     # 更新状态
#
#
#     def zero_grad(self):
#         super().zero_grad()
# class MetaAdamW(AdamW):
#     def __init__(self, net, *args, **kwargs):
#         super(MetaAdamW, self).__init__(*args, **kwargs)
#         self.net = net
#
#     def set_parameter(self, current_module, name, parameters):
#         if '.' in name:
#             name_split = name.split('.')
#             module_name = name_split[0]
#             rest_name = '.'.join(name_split[1:])
#             for children_name, children in current_module.named_children():
#                 if module_name == children_name:
#                     self.set_parameter(children, rest_name, parameters)
#                     break
#         else:
#             current_module._parameters[name] = parameters
#
#     def meta_step(self, grads):
#         assert len(list(self.net.parameters())) == len(grads), "Parameters and grads must match in length"
#         clip_grad_norm_(self.net.parameters(), max_norm=0.5)
#
#         for group in self.param_groups:
#             lr = group['lr']
#             weight_decay = group['weight_decay']
#             for (name, parameter), grad in zip(self.net.named_parameters(), grads):
#                 if grad is not None:
#                     if weight_decay != 0:
#                         grad_wd = grad.add(parameter.data, alpha=weight_decay)
#                     else:
#                         grad_wd = grad
#
#                     step_size = lr
#                     beta1, beta2 = group['betas']
#
#                     # Initialize state if not already done
#                     if 'exp_avg' not in self.state[parameter]:
#                         self.state[parameter]['exp_avg'] = torch.zeros_like(parameter.data)
#                     if 'exp_avg_sq' not in self.state[parameter]:
#                         self.state[parameter]['exp_avg_sq'] = torch.zeros_like(parameter.data)
#                     if 'step' not in self.state[parameter]:
#                         self.state[parameter]['step'] = 0
#
#                     exp_avg = self.state[parameter]['exp_avg']
#                     exp_avg_sq = self.state[parameter]['exp_avg_sq']
#                     step = self.state[parameter]['step'] + 1
#
#                     exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                     exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
#
#                     exp_avg_corr = exp_avg / (1 - beta1 ** step)
#                     exp_avg_sq_corr = exp_avg_sq / (1 - beta2 ** step)
#
#                     denom = exp_avg_sq_corr.sqrt().add(group['eps'])
#                     step_direction = exp_avg_corr / denom
#                     updated_param = parameter.data - step_size * step_direction
#
#                     self.state[parameter].update({
#                         "exp_avg": exp_avg,
#                         "exp_avg_sq": exp_avg_sq,
#                         "step": step
#                     })
#
#                     parameter.data.copy_(updated_param.add(grad_wd, alpha=-lr))
#                     self.set_parameter(self.net, name, parameter.add(grad_wd, alpha=-lr))
#                     # print(f'Updated parameter {name}: {parameter.data.shape}')  # Debugging line
#
#         # print('All parameters updated.')  # Debugging line
#
#     def zero_grad(self):
#         super().zero_grad()

class MetaAdamW(AdamW):
    def __init__(self, net, *args, **kwargs):
        super(MetaAdamW, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        assert len(list(self.net.parameters())) == len(grads), "Parameters and grads must match in length"
        clip_grad_norm_(self.net.parameters(), max_norm=0.5)

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            amsgrad = group.get('amsgrad', False)

            for (name, parameter), grad in zip(self.net.named_parameters(), grads):
                if grad is not None:
                    state = self.state[parameter]

                    # Initialize state if not already done
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(parameter.data)
                    if 'exp_avg_sq' not in state:
                        state['exp_avg_sq'] = torch.zeros_like(parameter.data)
                    if 'step' not in state:
                        state['step'] = 0  # initialize step as integer

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    step_t = state['step']

                    # Increment step
                    step_t += 1

                    # Perform step weight decay
                    parameter.data.mul_(1 - lr * weight_decay)

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias corrections
                    bias_correction1 = 1 - beta1 ** step_t
                    bias_correction2 = 1 - beta2 ** step_t

                    step_size = lr / bias_correction1

                    bias_correction2_sqrt = torch.sqrt(torch.tensor(bias_correction2, dtype=parameter.dtype))

                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        if 'max_exp_avg_sq' not in state:
                            state['max_exp_avg_sq'] = torch.zeros_like(parameter.data)
                        max_exp_avg_sq = state['max_exp_avg_sq']
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                    else:
                        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                    step_direction = exp_avg / denom

                    # Ensure shapes match for parameter update
                    if step_direction.shape != parameter.data.shape:
                        step_direction = step_direction.view_as(parameter.data)

                    updated_param = parameter.data - step_size * step_direction

                    # Update state
                    state['step'] = step_t

                    # Update the parameter using the optimized value
                    self.set_parameter(self.net, name, updated_param)

    def zero_grad(self):
        super().zero_grad()