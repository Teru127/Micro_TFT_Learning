from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Unit:
    hp: float
    atk: float
    armor: float

    @property
    def is_alive(self) -> bool:
        return self.hp > 0.0

@dataclass(frozen=True)
class BattleState:
    agent: Unit
    opp: Unit

    def to_observation(self) -> List[float]:
        "Convert into a flat list of floats for the RL agent to observe."
        return [
        self.agent.hp, self.opp.hp,
        self.agent.atk, self.opp.atk,
        self.agent.armor, self.opp.armor
        ]