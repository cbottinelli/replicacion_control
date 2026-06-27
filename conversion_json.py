import json
from dataclasses import asdict
from dataclasses import dataclass

@dataclass
class RatedModProposal:
    solution: str
    idea: str
    sneakyness_score: float
    guess_fail_rate_on_small_inps: float #esto es la salida de get_rarity()
    exploit_difficulty: float
    exploits: list[str]

@dataclass
class ProblemRatedModProposals:
    problem_id: str
    problem: str
    proposals: list[RatedModProposal]

def normalize_exploits(exploits):
    if exploits is None:
        return []
    if isinstance(exploits, list):
        return exploits
    if isinstance(exploits, str):
        return [exploits]
    raise TypeError(f"Unexpected exploits type: {type(exploits)}")


def problem_solution_json_to_problem_rated_mod_proposals(ps: dict) -> ProblemRatedModProposals:
    return ProblemRatedModProposals(
        problem_id=ps["problem_id"],
        problem=ps["problem"]["question"],
        proposals=[
            RatedModProposal(
                solution=p["backdoored_solution"],
                idea=p["idea"],
                sneakyness_score=p["sneakiness_score"],
                guess_fail_rate_on_small_inps=p["guess_fail_rate_on_small_inps"],
                exploit_difficulty=p["exploit_difficulty"],
                exploits=normalize_exploits(p["exploits"]),
            )
            for p in ps["backdoor_proposals"]
        ],
    )


with open("data/problem_solutions2.json", "r", encoding="utf-8") as f:
    old_problem_solutions = json.load(f)

problem_rated_mod_proposals = [
    problem_solution_json_to_problem_rated_mod_proposals(ps)
    for ps in old_problem_solutions
]

with open("data/problem_rated_mod_proposals2.json", "w", encoding="utf-8") as f:
    json.dump(
        [asdict(x) for x in problem_rated_mod_proposals],
        f,
        indent=2,
        ensure_ascii=False,
    )