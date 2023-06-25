from typing import Optional, List, Union
import torch


SCHEDULERS = {}

def register_scheduler(name: str):
    def wrapper(cls: GammaScheduler) -> GammaScheduler:
        if SCHEDULERS.get(name):
            raise NameError("Scheduler already has been registered.")
        SCHEDULERS[name] = cls
        return cls
    return wrapper

def get_gamma_scheduler(name: str, **kwargs):
    if SCHEDULERS.get(name, None) is None:
        raise NameError(f"Scheduler {name} does not exist.")
    return SCHEDULERS[name](**kwargs)


class GammaScheduler:
    def __init__(self,
                 gamma_src: Optional[float]=None,
                 gamma_dst: Optional[float]=None,
                 ratio: Optional[float]=None
                 ):

        self.gamma_src = gamma_src
        self.gamma_dst = gamma_dst
        self.ratio = ratio

    def get_gammas(self, num_time_steps: int) -> List[Union[float, torch.Tensor]]:
        raise NotImplementedError

    def __call__(self, num_time_steps: int) -> List[Union[float, torch.Tensor]]:
        return self.get_gammas(num_time_steps)


@register_scheduler(name='constant')
class ConstantGammaScheduler(GammaScheduler):
    def __init__(self, gamma_src: float):
        super().__init__(gamma_src=gamma_src)

    def get_gammas(self, num_time_steps: int) -> List[Union[float, torch.Tensor]]:
        return torch.tensor([self.gamma_src] * (num_time_steps+1))


@register_scheduler(name='step')
class StepGammaScheduler(GammaScheduler):
    def __init__(self, gamma_src: float, gamma_tau: float):
        super().__init__(gamma_src=gamma_src, ratio=gamma_tau)

    def get_gammas(self, num_time_steps: int) -> List[Union[float, torch.Tensor]]:
        nogamma_until = int(num_time_steps * self.ratio)
        gammas = torch.tensor([self.gamma_src] * (num_time_steps+1))
        gammas[ :nogamma_until] = 0.
        return gammas


@register_scheduler(name='reverse_step')
class ReverseStepGammaScheduler(GammaScheduler):
    def __init__(self, gamma_src: float, gamma_tau: float):
        super().__init__(gamma_src=gamma_src, ratio=gamma_tau)

    def get_gammas(self, num_time_steps: int) -> List[Union[float, torch.Tensor]]:
        turnoff_after = int(num_time_steps * self.ratio)
        gammas = torch.tensor([self.gamma_src] * (num_time_steps+1))
        gammas[turnoff_after: ] = 0.
        return gammas


@register_scheduler(name='linear')
class LinearGammaScheduler(GammaScheduler):
    def __init__(self, gamma_src: float, ratio: float):
        super().__init__(gamma_src=gamma_src, ratio=ratio)
        # this could be changed to use dst, not ratio.

    def get_gammas(self, num_time_steps: int) -> List[Union[float, torch.Tensor]]:
        dst = self.gamma_src + self.ratio * num_time_steps
        return torch.linspace(start=self.gamma_src,
                              end=dst,
                              steps=(num_time_steps+1))

@register_scheduler(name='exp')
class ExponentialGammaScheduler(GammaScheduler):
    def __init__(self, gamma_src: float, ratio: float):
        super().__init__(gamma_src=gamma_src, ratio=ratio)

    def get_gammas(self, num_time_steps: int) -> List[Union[float, torch.tensor]]:
        gammas = [self.gamma_src*self.ratio**i for i in range(num_time_steps+1)]
        return torch.tensor(gammas)

