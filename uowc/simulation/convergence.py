from dataclasses import dataclass
import math

@dataclass
class ConvergenceStatus:
    mean_power: float
    rel_error: float
    converged: bool
    n_batches: int

def power_rel_error(samples):
    n=len(samples)
    if n<2:
        return ConvergenceStatus(samples[-1] if samples else 0.0, float("inf"), False, n)
    mean=sum(samples)/n
    var=sum((x-mean)**2 for x in samples)/(n-1)
    se=math.sqrt(var/n)
    rel=se/abs(mean) if mean!=0 else float("inf")
    return ConvergenceStatus(mean, rel, False, n)
