from typing import Optional, Dict
from .duck_basemodel import DuckDBModel


class Message(DuckDBModel):
    run_id: int
    id: Optional[int] = None
    cpu_usage: float
    gpu_usage: Optional[float] = None
    disk_usage: float
    prompt: str
    response: str

    @classmethod
    def create_llm_message(
        cls,
        run_id: int,
        prompt: str,
        response: str,
        power_usage: Dict[str, float],
    ) -> "Message":
        """
        Create a new message record for an LLM interaction with power usage metrics.

        Args:
            run_id: ID of the associated run
            prompt: Input prompt sent to the LLM
            response: Response received from the LLM
            power_usage: Dictionary containing power usage metrics
                Expected format:
                {
                    'cpu': float,     # CPU power usage
                    'dram': float,    # DRAM power usage
                    'gpu': float,     # GPU power usage
                    'disk': float     # Disk power usage
                }

        Returns:
            Message: A new Message instance that has been saved to the database

        Example:
            power_data = {
                'cpu': 241.306999,
                'dram': 0.0,
                'gpu': 1189.9352091659741,
                'disk': 0.00031072875
            }

            message = Message.create_llm_message(
                run_id=1,
                prompt="What is the capital of France?",
                response="The capital of France is Paris.",
                power_usage=power_data
            )
        """
        # Extract CPU power usage (takes first value if it's an array)
        cpu_usage = power_usage["cpu"]
        if hasattr(cpu_usage, "__iter__"):
            cpu_usage = cpu_usage[0]

        # Create the message instance
        message = cls(
            run_id=run_id,
            prompt=prompt,
            response=response,
            cpu_usage=cpu_usage,
            gpu_usage=power_usage["gpu"],
            disk_usage=power_usage["disk"],
        )

        message.save()
        return message
