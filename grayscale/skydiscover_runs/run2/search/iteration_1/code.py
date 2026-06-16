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

    DIVERGE_LABEL = "diverge"
    REFINE_LABEL = "refine"

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self._sample_count = 0

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

    def _score_of(self, program) -> float:
        """Extract a comparable scalar score from a program, defaulting to 0.0."""
        for attr in ("combined_score", "score"):
            val = getattr(program, attr, None)
            if isinstance(val, (int, float)):
                return float(val)
        metrics = getattr(program, "metrics", None)
        if isinstance(metrics, dict):
            for key in ("combined_score", "score"):
                v = metrics.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
        return 0.0

    def sample(
        self,
        num_inspirations: Optional[int] = 4,
        **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Fitness-aware sampling that alternates between REFINE (exploit the best
        programs) and DIVERGE (explore broadly) modes.

        Strategy:
          - Parent: usually selected with a strong bias toward high-scoring
            programs (rank-weighted), with elitism occasionally picking the best,
            and a small chance of pure-random exploration.
          - Inspirations: a curated mix of top performers (to teach the model
            what good solutions look like) and diverse random picks (to inject
            variety and avoid premature convergence).
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        self._sample_count += 1

        # Sort candidates by score descending (best first).
        scored = sorted(candidates, key=self._score_of, reverse=True)
        n = len(scored)

        # Decide mode: alternate exploitation (refine) vs exploration (diverge).
        # Bias toward refinement since exploiting known-good solutions tends to
        # yield steady improvements once a good region is found.
        refine_mode = random.random() < 0.7
        mode_label = self.REFINE_LABEL if refine_mode else self.DIVERGE_LABEL

        # ----- Parent selection -----
        r = random.random()
        if refine_mode:
            if r < 0.30:
                # Elitism: refine the current best directly.
                parent = scored[0]
            elif r < 0.85:
                # Rank-weighted selection biased toward the top of the pool.
                # Quadratic weighting strongly favors higher-ranked programs.
                weights = [(n - i) ** 2 for i in range(n)]
                parent = random.choices(scored, weights=weights, k=1)[0]
            else:
                # Occasional broad pick to escape local optima.
                parent = random.choice(scored)
        else:
            # Diverge: explore by sampling more uniformly, slightly favoring
            # mid/lower-rank programs to probe under-explored regions.
            if r < 0.5:
                parent = random.choice(scored)
            else:
                # Pick from the less-exploited bottom half occasionally.
                lower = scored[n // 2:] if n > 1 else scored
                parent = random.choice(lower)

        # ----- Inspiration selection -----
        pool = [p for p in scored if p.id != parent.id]
        k = min(num_inspirations or 0, len(pool))

        examples: List[EvolvedProgram] = []
        chosen_ids = set()

        if k > 0:
            # Always include a couple of top performers as quality anchors.
            num_top = max(1, k // 2) if refine_mode else max(1, k // 3)
            for p in pool[:num_top]:
                if p.id not in chosen_ids:
                    examples.append(p)
                    chosen_ids.add(p.id)
                if len(examples) >= num_top:
                    break

            # Fill the rest with diverse random picks for variety.
            remaining_pool = [p for p in pool if p.id not in chosen_ids]
            random.shuffle(remaining_pool)
            for p in remaining_pool:
                if len(examples) >= k:
                    break
                examples.append(p)
                chosen_ids.add(p.id)

            examples = examples[:k]

        parent_dict = {mode_label: parent}
        inspiration_programs_dict = {mode_label: examples}

        return parent_dict, inspiration_programs_dict

# EVOLVE-BLOCK-END
