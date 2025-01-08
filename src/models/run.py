from datetime import datetime, timedelta
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

    def formatted_name(self):
        xs = self.model_name.split("-")
        name = "-".join(xs[0:3]).split(".Q4_K_M.gguf")[0]

        if name == "rf":
            name = "Random Forest"
        elif name == "svm":
            name = "Support Vector Machine"
        else:
            name = name

        return f"{self.id}: {name}"

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

    def get_end_time(self) -> Optional[datetime]:
        """
        Get the timestamp of the last message in this run.

        Returns:
            datetime | None: Timestamp of the last message or None if no messages exist
        """
        messages = self.get_messages()
        return max(msg.created_at for msg in messages) + timedelta(hours=1)

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
        ax1.set_xlabel("Message Number")
        ax1.set_ylabel("Energy Usage (J)")
        ax1.set_title(f"Energy Usage Over Time - Run {self.id} ({self.model_name})")
        ax1.legend()
        ax1.grid(True)

        # 2. Average energy usage bar chart
        ax2 = fig.add_subplot(gs[1, 0])
        components = ["CPU", "GPU"]
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
            (self.get_end_time() - self.start_time).total_seconds()
            if self.get_end_time()
            else None
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
            total_energy = msg.cpu_usage + (msg.gpu_usage or 0)

            conversation_data.append(
                {
                    "Timestamp": msg.created_at,
                    "Prompt": msg.prompt,
                    "Response": msg.response,
                    "Metadata": msg.metadata,
                }
            )

        return pd.DataFrame(conversation_data)

    def get_power_stats_df(self) -> pd.DataFrame:
        """
        Get summary statistics of power and energy usage for this run as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing energy and power statistics for CPU, GPU
                with columns for mean, max, total energy, and percentage of total energy.
                If the run is completed, also includes average power consumption.

        Raises:
            ValueError: If no messages are found for this run
        """
        messages = self.get_messages()
        if not messages:
            raise ValueError(f"No messages found for run {self.id}")

        # Convert messages to DataFrame for analysis
        data = pd.DataFrame(
            [
                {
                    "cpu_energy": msg.cpu_usage,
                    "gpu_energy": msg.gpu_usage if msg.gpu_usage is not None else 0,
                    "timestamp": msg.created_at,
                }
                for msg in messages
            ]
        )

        # Calculate total energy for percentage calculations
        total_system_energy = data["cpu_energy"].sum() + data["gpu_energy"].sum()

        # Calculate statistics for each component
        stats_dict = {
            "Component": [
                "CPU",
                "GPU",
            ],
            "Mean Energy (J)": [
                data["cpu_energy"].mean(),
                data["gpu_energy"].mean(),
            ],
            "Max Energy (J)": [
                data["cpu_energy"].max(),
                data["gpu_energy"].max(),
            ],
            "Total Energy (J)": [
                data["cpu_energy"].sum(),
                data["gpu_energy"].sum(),
            ],
            "Energy Percentage (%)": [
                (data["cpu_energy"].sum() / total_system_energy) * 100,
                (data["gpu_energy"].sum() / total_system_energy) * 100,
            ],
        }

        # Calculate duration and average power if run is completed
        if self.end_time is not None:
            duration = (self.get_end_time() - self.start_time).total_seconds()
            stats_dict["Average Power (W)"] = [
                data["cpu_energy"].sum() / duration,
                data["gpu_energy"].sum() / duration,
            ]

        # Create DataFrame with all statistics
        stats_df = pd.DataFrame(stats_dict)

        # Add system total row
        total_row = pd.DataFrame(
            {
                "Component": ["System Total"],
                "Mean Energy (J)": [
                    data[["cpu_energy", "gpu_energy"]].sum(axis=1).mean()
                ],
                "Max Energy (J)": [
                    data[["cpu_energy", "gpu_energy"]].sum(axis=1).max()
                ],
                "Total Energy (J)": [total_system_energy],
                "Energy Percentage (%)": [100.0],
            }
        )

        # Add average power for system total if run is completed
        if self.get_end_time() is not None:
            total_row["Average Power (W)"] = [total_system_energy / duration]

        stats_df = pd.concat([stats_df, total_row], ignore_index=True)

        # Round numeric columns to 2 decimal places
        numeric_columns = stats_df.select_dtypes(include=["float64"]).columns
        stats_df[numeric_columns] = stats_df[numeric_columns].round(2)

        return stats_df
