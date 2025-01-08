from abc import abstractmethod
from typing import List, Dict, Any


class BaseWrapper:
    _instance = None
    priority = 1000
    allowed_kwargs = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BaseWrapper, cls).__new__(cls)
        return cls._instance

    @abstractmethod
    def process_audio(self, inputs: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        pass

    @abstractmethod
    def register_api_endpoint(self, api) -> Any:
        pass
