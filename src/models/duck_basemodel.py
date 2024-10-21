from abc import ABC, abstractmethod
import duckdb
from pydantic import BaseModel
from typing import ClassVar, List, Type, TypeVar, Any

T = TypeVar("T", bound="DuckDBModel")


class DuckDBModel(BaseModel, ABC):
    _db_connection: ClassVar[duckdb.DuckDBPyConnection] = None
    _table_name: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_row(cls: Type[T], row: tuple) -> T:
        """Create an instance of the model from a database row."""
        pass

    def to_row(self) -> tuple:
        """Convert the instance to a tuple for database insertion."""
        return tuple(
            getattr(self, field)
            for field in self.__annotations__
            if field not in self.__class__.__class_vars__
        )

    @classmethod
    def initialize_db(cls, db_path: str):
        """Initialize the database connection."""
        if cls._db_connection is None:
            cls._db_connection = duckdb.connect(db_path)

        # Verify that the table exists
        table_exists = cls._db_connection.execute(
            f"""
            SELECT name FROM sqlite_master WHERE type='table' AND name='{cls._table_name}';
        """
        ).fetchone()

        if not table_exists:
            raise RuntimeError(
                f"The '{cls._table_name}' table does not exist in the database."
            )

    @classmethod
    def close_db(cls):
        """Close the database connection."""
        if cls._db_connection:
            cls._db_connection.close()
            cls._db_connection = None

    def save(self):
        """Save the current instance to the database."""
        if self._db_connection is None:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        columns = ", ".join(
            field
            for field in self.__annotations__
            if field not in self.__class__.__class_vars__
        )
        placeholders = ", ".join(["?" for _ in columns.split(", ")])

        self._db_connection.execute(
            f"""
            INSERT INTO {self._table_name} ({columns})
            VALUES ({placeholders})
        """,
            self.to_row(),
        )

    @classmethod
    def get_all(cls: Type[T]) -> List[T]:
        """Retrieve all rows from the database."""
        if cls._db_connection is None:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        results = cls._db_connection.execute(
            f"SELECT * FROM {cls._table_name}"
        ).fetchall()
        return [cls.from_row(row) for row in results]

    @classmethod
    def get_by_field(cls: Type[T], field: str, value: Any) -> List[T]:
        if cls._db_connection is None:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        # Get the column type
        column_info = cls._db_connection.execute(
            f"PRAGMA table_info({cls._table_name})"
        ).fetchall()
        column_type = next((col[2] for col in column_info if col[1] == field), None)

        # Convert value if necessary
        if column_type == "INTEGER" and not isinstance(value, int):
            try:
                value = int(value)
            except ValueError:
                raise ValueError(
                    f"Cannot convert '{value}' to integer for field '{field}'"
                )

        results = cls._db_connection.execute(
            f"SELECT * FROM {cls._table_name} WHERE {field} = ?", [value]
        ).fetchall()
        return [cls.from_row(row) for row in results]
