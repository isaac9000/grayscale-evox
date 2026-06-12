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
        Selects parent preferring high-scoring programs via tournament selection,
        and inspirations via diverse sampling weighted by score to guide search
        toward high-combined-score solutions while maintaining diversity.
        """
        candidates = list(self.programs.values())
        
        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        # Tournament selection for parent: pick best of k random candidates
        tournament_size = min(3, len(candidates))
        tournament = random.sample(candidates, tournament_size)
        parent = max(tournament, key=lambda p: getattr(p, 'combined_score', 0.0) or 0.0)

        # For inspirations, prefer diversity: sample broadly but weight toward high scores
        sample_size = min(num_inspirations * 2 + 1, len(candidates))
        pool = random.sample(candidates, sample_size)
        pool = [p for p in pool if p.id != parent.id]

        # Sort by score descending, take top num_inspirations for quality guidance
        pool.sort(key=lambda p: getattr(p, 'combined_score', 0.0) or 0.0, reverse=True)
        examples = pool[:num_inspirations]

        parent_dict = {"": parent}
        inspiration_programs_dict = {"": examples}

        return parent_dict, inspiration_programs_dict

# EVOLVE-BLOCK-END
