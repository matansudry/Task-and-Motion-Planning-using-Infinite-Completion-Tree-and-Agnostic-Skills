from typing import Literal, Optional, Dict
import os

from functools import cached_property
from dataclasses import dataclass

@dataclass
class PDDLConfig:
    domain_dir: str = "configs/pybullet/envs/official/template"
    domain_file: str = "template_valid_domain.pddl"
    problem_dir: Optional[str] = None
    problem_subdir: Optional[str] = None
    instruction_dir: Optional[str] = None
    prompt_dir: Optional[str] = None
    custom_domain_file: Optional[str] = None
    _pddl_problem_file: Optional[str] = None

    @cached_property
    def pddl_domain_file(self) -> str:
        if self.custom_domain_file is not None:
            return self.custom_domain_file
        return os.path.join(self.domain_dir, self.domain_file)

    @property
    def pddl_problem_file(self) -> str:
        if self._pddl_problem_file is None:
            raise ValueError("Must set PDDLConfig.pddl_problem_file before calling.")
        return self._pddl_problem_file

    @pddl_problem_file.setter
    def pddl_problem_file(self, x: str) -> None:
        self._pddl_problem_file = x

    @property
    def pddl_domain_dir(self) -> str:
        return self.domain_dir

    @property
    def pddl_problem_dir(self) -> str:
        if self.problem_dir is None and self.problem_subdir is None:
            problem_dir = self.domain_dir
        elif self.problem_dir is None and self.problem_subdir is not None:
            problem_dir = os.path.join(self.domain_dir, self.problem_subdir)
        else:
            problem_dir = self.problem_dir
        return problem_dir

    def get_problem_file(self, problem_name: str) -> str:
        problem_dir = self.pddl_problem_dir
        problem_file = os.path.join(problem_dir, problem_name + ".pddl")
        self.pddl_problem_file = problem_file
        return problem_file

    def get_instructions_file(self, problem_name: str) -> str:
        if self.instruction_dir is not None:
            return os.path.join(
                self.instruction_dir, problem_name + "_instructions.txt"
            )
        return os.path.join(self.domain_dir, problem_name + "_instructions.txt")

    def get_prompt_file(self, problem_name: str) -> str:
        if self.prompt_dir is not None:
            return os.path.join(self.prompt_dir, problem_name + "_prompt.json")
        return os.path.join(self.domain_dir, problem_name + "_prompt.json")