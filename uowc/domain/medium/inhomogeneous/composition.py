
from dataclasses import dataclass
@dataclass
class InhomogeneousMedium:
    layered: object|None=None
    gradient: object|None=None
    turbulence: object|None=None
