from abc import ABC, abstractmethod


class Reward(ABC):
    @abstractmethod
    def compute(self, prompt: str, response: str, gold: dict) -> float:
        """Returns a scalar reward in [0, 1]."""
        pass
