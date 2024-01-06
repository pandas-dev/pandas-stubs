from collections.abc import Hashable
import dataclasses

@dataclasses.dataclass(order=True, frozen=True)
class OutputKey:
    label: Hashable
    position: int

plotting_methods: frozenset[str]
cythonized_kernels: frozenset[str]
reduction_kernels: frozenset[str]
transformation_kernels: frozenset[str]
groupby_other_methods: frozenset[str]
transform_kernel_allowlist: frozenset[str]
