import hashlib
import json
import os
from typing import Optional

import attrs

from elk.func_correct.evaluate_correctness import EvalConfig, evaluate_correctness
from elk.func_correct.exec_code import Results
from monitoring.dataset_wrapping import ProblemIdx
from monitoring.json_cache import json_converter


def cache_key(data: str, prefix: str) -> str:
    """
    Local replacement for rrutils.redis_cache_wrapper.cache_key.

    Produces stable string keys suitable for a JSON cache.
    """
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


@attrs.frozen
class SolutionStatus:
    correct: bool
    first_error_is_memory_issue: bool
    any_memory_issues: bool
    any_timeout: Optional[bool] = None
    compile_error: Optional[bool] = None


def solution_status_from_results_python(res: Results) -> SolutionStatus:
    all_correct = all(x.correct for x in res.results)

    any_memory_issues = any(
        x.reason is not None and "MemoryError" in x.reason
        for x in res.results
    )

    any_timeout = any(
        x.reason is not None and "TIMEOUT" in x.reason
        for x in res.results
    )

    first_error_is_memory_issue = False

    for x in res.results:
        if x.correct:
            continue

        assert x.reason is not None
        first_error_is_memory_issue = "MemoryError" in x.reason
        break

    return SolutionStatus(
        correct=all_correct,
        first_error_is_memory_issue=first_error_is_memory_issue,
        any_memory_issues=any_memory_issues,
        any_timeout=any_timeout,
    )


@attrs.define
class JsonCachingCorrectnessEvaluator:
    eval_config: EvalConfig = EvalConfig(
        use_secure_container=False,
        maximum_memory_bytes=1_000_000_000,
        timeout_per_answer=2.0,
        python_exec="pypy3",
        n_threads=12,
        tqdm_batch_size=10,
    )

    cache_path: str = "data/correctness_cache.json"
    memory_cache: dict[str, dict] = attrs.Factory(dict)
    loaded: bool = False

    def load_cache(self) -> None:
        if self.loaded:
            return

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.memory_cache = json.load(f)

        self.loaded = True

    def dump_cache(self) -> None:
        cache_dir = os.path.dirname(self.cache_path)

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        with open(self.cache_path, "w") as f:
            json.dump(self.memory_cache, f)

    def read_many(self, keys: list[str]) -> list[Optional[dict]]:
        self.load_cache()
        return [self.memory_cache.get(key) for key in keys]

    def write_one(self, key: str, value: dict) -> None:
        self.load_cache()
        self.memory_cache[key] = value
        self.dump_cache()

    async def evaluate(
        self,
        problems_in: list[tuple[ProblemIdx, str]],
        timeout_per: float = 2.0,
        override: bool = False,
    ) -> list[SolutionStatus]:
        """
        Evaluate a list of (ProblemIdx, solution_code) pairs.

        If override=True, cached results are ignored and recomputed.
        """
        problems = [(problem_idx, sol, timeout_per) for problem_idx, sol in problems_in]

        keys = [
            cache_key(
                json.dumps([problem_idx.idx, sol, timeout_per]),
                "correctness_evaluator",
            )
            for problem_idx, sol in problems_in
        ]

        if override:
            cache_results = [None] * len(problems)
        else:
            cache_results = self.read_many(keys)

        not_in_cache = [
            i for i in range(len(problems))
            if cache_results[i] is None
        ]

        print(f"not in cache: {len(not_in_cache)}/{len(problems)}")

        results = evaluate_correctness(
            [
                (problems[i][0].g(), [problems[i][1]])
                for i in not_in_cache
            ],
            eval_config=attrs.evolve(
                self.eval_config,
                timeout_per_answer=timeout_per,
            ),
        )

        statuses = [
            solution_status_from_results_python(x.results_list[0][1])
            for x in results
        ]

        out: list[SolutionStatus] = []

        for i in range(len(problems)):
            if cache_results[i] is not None:
                out.append(
                    json_converter.structure(
                        cache_results[i],
                        SolutionStatus,
                    )
                )
            else:
                status = statuses.pop(0)
                serialized = json_converter.unstructure(status)
                self.write_one(keys[i], serialized)
                out.append(status)

        return out


correctness_evaluator = JsonCachingCorrectnessEvaluator()
