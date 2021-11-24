from typing import List, Tuple, Dict, Optional, Iterable, Type, Union, Iterator
from collections import OrderedDict
import sys

import torch
import torch.nn as nn


class LazyModuleList(nn.Module):
    """Lazy Implementation of nn.ModuleList"""

    def __init__(
        self,
        modules_defs: Optional[Iterable[Tuple[Type[nn.Module], List, Dict]]] = None,
        modules_checkpoints: Optional[Iterable[str]] = None,
        max_instantied: int = 1,
        check_refs: bool = True,
        delete_schedule: str = "oldest"
):
        super().__init__()
        #if preload_modules:
        #    assert delete_schedule == "oldest", "preloading only works with a FIFO schedule"

        self.preload_modules = delete_schedule == "oldest"

        self.max_instantied = max_instantied
        self.instantied_modules = OrderedDict()
        self.check_refs = check_refs
        self.delete_schedule = delete_schedule

        if modules_defs is not None:
            self.modules_defs = list(modules_defs)
            self.modules_checkpoints = (
                list(modules_checkpoints) if modules_checkpoints is not None else None
            )

    def load_module(self, idx: int):
        if len(self.instantied_modules) >= self.max_instantied:
            # get the oldest/newest module instantied
            ordered_idxs = list(iter(self.instantied_modules.keys()))
            if self.delete_schedule == "oldest":
                idx_to_delete = ordered_idxs[0]
            elif self.delete_schedule == "newest":
                idx_to_delete = ordered_idxs[-1]
            else:
                raise ValueError("delete_schedule must be either 'oldest' or 'newest'")

            # check for reference count to make sure there are no 
            # outside references that could prevent the deletion of the object
            # since there is temp reference created by `sys.getrefcount`
            # plus the 2 refences created by `self.instantied_modules`
            # we check for 4 rather than 1 
            # https://docs.python.org/3.8/library/sys.html#sys.getrefcount
            # TODO: check if its really `self.instantied_modules` that holds 2 refs
            
            if self.check_refs and sys.getrefcount(self.instantied_modules[idx_to_delete]) > 4:
                print(
                    "warning: module is being referenced outside the lazy wrapper..."
                )
                print(
                    "it is very likely it will not be deleted, causing unexpected memory usage"
                )

            del self.instantied_modules[idx_to_delete]
            

        module_cls, module_args, module_kwargs = self.modules_defs[idx]
        self.instantied_modules[idx] = module_cls(*module_args, **module_kwargs)
        if self.modules_checkpoints is not None:
            self.instantied_modules[idx].load_state_dict(
                torch.load(self.modules_checkpoints[idx])
            )

    def __getitem__(self, idx: int) -> nn.Module:
        # TODO: add slices
        if idx not in self.instantied_modules:
            self.load_module(idx)
            if self.preload_modules:
                for next_idxs in range(idx + 1, min(idx+self.max_instantied, len(self.modules_defs))):
                    self.load_module(next_idxs)
        return self.instantied_modules[idx]

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        # TODO: fix this
        raise NotImplementedError("")

    def __delitem__(self, idx: Union[int, slice]) -> None:
        # TODO: add slices
        # TODO: fix this to delete references in `instantiated_modules`
        raise NotImplementedError("")

    def __len__(self) -> int:
        return len(self.module_defs)


# TODO: add decorator that replaces ModuleList
# with LazyModuleList automatically
# not trivial, will involve complex parsing of code
# checking for constructor calls

# TODO: based on the previous inspector, add something that automatically creates
# the needed checkpoint division algorithm
