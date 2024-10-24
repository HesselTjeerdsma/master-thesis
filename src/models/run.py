from datetime import datetime
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .duck_basemodel import DuckDBModel
from .message import Message  # Import the Message model


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

    def get_messages(self) -> List[Message]:
        """
        Retrieve all messages associated with this run.

        Returns:
            List[Message]: List of Message objects for this run ordered by ID
        """
        return Message.list_all_filtered("run_id", self.id)

    def plot_power_usage(self, save_path: Optional[str] = None) -> None:
        """
        Create visualizations of energy usage for this run using matplotlib.

        Args:
            save_path: Optional path to save the plot (e.g., 'energy_usage.png')

        Returns:
            None - displays or saves the plot
        """
        messages = self.get_messages()
        if not messages:
            raise ValueError(f"No messages found for run {self.id}")

        # Convert messages to DataFrame for easier analysis
        data = pd.DataFrame(
            [
                {
                    "cpu_energy": msg.cpu_usage,
                    "gpu_energy": msg.gpu_usage if msg.gpu_usage is not None else 0,
                    "disk_energy": msg.disk_usage,
                    "message_num": i + 1,
                    "timestamp": msg.created_at,
                }
                for i, msg in enumerate(messages)
            ]
        )

        # Calculate statistics
        stats = {
            "CPU": {
                "mean": data["cpu_energy"].mean(),
                "max": data["cpu_energy"].max(),
                "total": data["cpu_energy"].sum(),
            },
            "GPU": {
                "mean": data["gpu_energy"].mean(),
                "max": data["gpu_energy"].max(),
                "total": data["gpu_energy"].sum(),
            },
            "Disk": {
                "mean": data["disk_energy"].mean(),
                "max": data["disk_energy"].max(),
                "total": data["disk_energy"].sum(),
            },
        }

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))  # Increased height to accommodate text
        gs = fig.add_gridspec(
            3, 2, height_ratios=[2, 1, 0.5]
        )  # Added third row for text

        # 1. Time series of all components
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(data["message_num"], data["cpu_energy"], label="CPU", marker="o")
        ax1.plot(data["message_num"], data["gpu_energy"], label="GPU", marker="o")
        ax1.plot(data["message_num"], data["disk_energy"], label="Disk", marker="o")
        ax1.set_xlabel("Message Number")
        ax1.set_ylabel("Energy Usage (J)")
        ax1.set_title(f"Energy Usage Over Time - Run {self.id} ({self.model_name})")
        ax1.legend()
        ax1.grid(True)

        # 2. Average energy usage bar chart
        ax2 = fig.add_subplot(gs[1, 0])
        components = ["CPU", "GPU", "Disk"]
        means = [stats[comp]["mean"] for comp in components]
        bars = ax2.bar(components, means)
        ax2.set_ylabel("Average Energy Usage (J)")
        ax2.set_title("Average Energy Usage by Component")
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}J",
                ha="center",
                va="bottom",
            )

        # 3. Energy distribution pie chart
        ax3 = fig.add_subplot(gs[1, 1])
        total_energy = sum(stats[comp]["total"] for comp in components)
        sizes = [stats[comp]["total"] for comp in components]
        percentages = [size / total_energy * 100 for size in sizes]
        ax3.pie(
            percentages,
            labels=[
                f"{comp}\n({perc:.1f}%)" for comp, perc in zip(components, percentages)
            ],
            autopct="%1.1f%%",
        )
        ax3.set_title("Total Energy Distribution")

        # Add summary statistics and run information as text in the bottom row
        duration = (
            (self.end_time - self.start_time).total_seconds() if self.end_time else None
        )
        duration_text = (
            f"Duration: {duration:.1f}s" if duration else "Duration: Running"
        )

        # Calculate average power (energy/time) if duration is available
        avg_power_text = ""
        if duration:
            avg_power = total_energy / duration
            avg_power_text = f"Average Power: {avg_power:.2f}W"

        # Create text axes in the bottom row
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis("off")

        run_info = (
            f"Run Information:\n"
            f"Model: {self.model_name} | Environment: {self.environment} | "
            f"Status: {self.status} | {duration_text} | {avg_power_text}\n\n"
            f"Summary Statistics:\n"
            f"Messages: {len(messages)} | "
            f"CPU: Mean={stats['CPU']['mean']:.2f}J, Max={stats['CPU']['max']:.2f}J | "
            f"GPU: Mean={stats['GPU']['mean']:.2f}J, Max={stats['GPU']['max']:.2f}J | "
            f"Disk: Mean={stats['Disk']['mean']:.2f}J, Max={stats['Disk']['max']:.2f}J | "
            f"Total Energy: {total_energy:.2f}J"
        )

        ax_text.text(
            0.5,
            0.5,
            run_info,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def get_conversation_df(self) -> pd.DataFrame:
        """
        Create a DataFrame showing the conversation history with prompts and responses.

        Returns:
            pd.DataFrame: DataFrame containing message number, prompt, response, and energy metrics
        """
        messages = self.get_messages()
        if not messages:
            raise ValueError(f"No messages found for run {self.id}")

        # Create DataFrame with conversation history and energy metrics
        conversation_data = []
        for i, msg in enumerate(messages, 1):
            total_energy = msg.cpu_usage + (msg.gpu_usage or 0) + msg.disk_usage

            conversation_data.append(
                {
                    "Timestamp": msg.created_at,
                    "Prompt": msg.prompt,
                    "Response": msg.response,
                }
            )

        return pd.DataFrame(conversation_data)
