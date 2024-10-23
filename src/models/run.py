from datetime import datetime
from typing import Optional, Dict
from .duck_basemodel import DuckDBModel


class Run(DuckDBModel):
    id: Optional[int] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    model_name: str
    environment: str
    status: str
    metadata: Optional[Dict] = None

    @classmethod
    def start(
        cls, model_name: str, environment: str, metadata: Optional[Dict] = None
    ) -> "Run":
        """
        Create and start a new run with the current timestamp.

        Args:
            model_name: Name of the model being run
            environment: Environment where the run is executing
            metadata: Optional dictionary of additional run metadata

        Returns:
            Run: A new Run instance that has been saved to the database
        """
        run = cls(
            start_time=datetime.now(),
            model_name=model_name,
            environment=environment,
            status="running",
            metadata=metadata,
        )
        run.save()
        return run

    def end(self, status: str = "completed") -> None:
        """
        End the current run with the specified status and current timestamp.

        Args:
            status: Final status of the run (default: "completed")
                   Common values: "completed", "failed", "cancelled"

        Raises:
            ValueError: If the run is already ended
        """
        if self.end_time is not None:
            raise ValueError(
                f"Run {self.id} is already ended with status: {self.status}"
            )

        self.end_time = datetime.now()
        self.status = status
        self.save()
