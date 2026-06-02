
from .homogeneous import HomogeneousMedium
from .inhomogeneous.composition import InhomogeneousMedium

def build_medium(kind="homogeneous", **kwargs):
    if kind=="homogeneous":
        return HomogeneousMedium(**kwargs)
    return InhomogeneousMedium(**kwargs)
