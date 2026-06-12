# EVOLVE-BLOCK-START
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.config import DatabaseConfig
from typing import Optional, Tuple, List, Dict
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""
    

class EvolvedProgramDatabase(ProgramDatabase):
    """Initial search strategy database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

        if config.random_seed is not None:
            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, 
        num_inspirations: Optional[int] = 4,
        **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Uniform random parent and inspiration selection for maximum diversity.
        Simple random sampling across all candidates without score bias.
        """
        candidates = list(self.programs.values())
        
        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        parent = random.choice(candidates)

        sample_size = min(num_inspirations + 1, len(candidates))
        examples = random.sample(candidates, sample_size)
        examples = [p for p in examples if p.id != parent.id][:num_inspirations]

        parent_dict = {"": parent}
        inspiration_programs_dict = {"": examples}

        return parent_dict, inspiration_programs_dict

# EVOLVE-BLOCK-END
