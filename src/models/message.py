# Example usage with the Message model
from typing import ClassVar, Optional, Dict
from .duck_basemodel import DuckDBModel
import numpy as np


class Message(DuckDBModel):
    run_id: str
    message_id: str
    cpu_usage: float
    gpu_usage: Optional[float] = None
    disk_usage: float

    _table_name: ClassVar[str] = "messages"

    @classmethod
    def from_row(cls, row: tuple) -> "Message":
        return cls(
            **dict(
                zip(
                    ["run_id", "message_id", "cpu_usage", "gpu_usage", "disk_usage"],
                    row,
                )
            )
        )

    @classmethod
    def from_dict(cls, data: Dict[str, any], run_id: str, message_id: str) -> "Message":
        """
        Create a Message instance from a dictionary of resource usage data.

        Args:
            data (Dict[str, any]): Dictionary containing resource usage data.
            run_id (str): The run ID for this message.
            message_id (str): The message ID for this instance.

        Returns:
            Message: A new Message instance.
        """
        return cls(
            run_id=run_id,
            message_id=message_id,
            cpu_usage=float(np.mean(data["cpu"])),  # Taking mean if it's an array
            gpu_usage=float(data["gpu"]) if "gpu" in data else None,
            disk_usage=float(data["disk"]),
        )

    def total_resource_usage(self) -> float:
        """Calculate the total resource usage."""
        return self.cpu_usage + (self.gpu_usage or 0) + self.disk_usage

    def is_gpu_active(self) -> bool:
        """Check if GPU is being used."""
        return self.gpu_usage is not None and self.gpu_usage > 0
