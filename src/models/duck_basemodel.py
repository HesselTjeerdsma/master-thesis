from typing import Dict, List, Optional, Type, TypeVar, Any
import duckdb
from datetime import datetime
import json
from pydantic import BaseModel

T = TypeVar("T", bound="DuckDBModel")


class DuckDBModel(BaseModel):
    """
    Abstract base class for DuckDB models with automatic ID handling
    and basic CRUD operations.
    """

    _connection: Optional[duckdb.DuckDBPyConnection] = None
    _table_name: Optional[str] = None

    @classmethod
    def initialize_db(cls, path: str = ":memory:") -> None:
        """Initialize DuckDB connection and create tables for all model classes."""
        cls._connection = duckdb.connect(path)

        # Create tables for all subclasses
        for subclass in cls.__subclasses__():
            table_name = subclass.__name__.lower()
            subclass._table_name = table_name

            # Create sequence for ID
            sequence_name = f"seq_{table_name}_id"
            cls._connection.execute(
                f"CREATE SEQUENCE IF NOT EXISTS {sequence_name} START 1"
            )

            # Generate CREATE TABLE statement based on model fields
            columns = []
            for field_name, field in subclass.__annotations__.items():
                if field_name.startswith("_"):
                    continue

                # Map Python types to SQL types
                field_type = str(field)
                sql_type = cls._get_sql_type(field_type)

                # Handle Optional types
                is_optional = "Optional" in field_type
                nullable = "NULL" if is_optional else "NOT NULL"

                # Special handling for ID field
                if field_name == "id":
                    columns.append(f"id INTEGER PRIMARY KEY")
                    continue

                columns.append(f"{field_name} {sql_type} {nullable}")

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)}
            )
            """
            cls._connection.execute(create_table_sql)

    @staticmethod
    def _get_sql_type(python_type: str) -> str:
        """Map Python types to SQL types."""
        type_mapping = {
            "int": "INTEGER",
            "float": "DOUBLE",
            "str": "VARCHAR",
            "bool": "BOOLEAN",
            "datetime": "TIMESTAMP",
            "Dict": "JSON",
            "dict": "JSON",
            "Optional[int]": "INTEGER",
            "Optional[float]": "DOUBLE",
            "Optional[str]": "VARCHAR",
            "Optional[datetime]": "TIMESTAMP",
            "Optional[Dict]": "JSON",
            "Optional[dict]": "JSON",
        }
        for py_type, sql_type in type_mapping.items():
            if py_type in python_type:
                return sql_type
        return "VARCHAR"  # Default to VARCHAR for unknown types

    def save(self) -> None:
        """Save the model instance to the database."""
        if not self._connection or not self._table_name:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        # Convert model to dict, excluding None values and handling special types
        data = {k: v for k, v in self.dict().items() if v is not None and k != "id"}

        # Handle JSON fields
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = json.dumps(value)
            elif isinstance(value, datetime):
                data[key] = value.isoformat()

        # Generate SQL for insert/update
        if not hasattr(self, "id") or self.id is None:
            # Insert new record using sequence
            fields = list(data.keys())
            placeholders = [f"${i+1}" for i in range(len(fields))]
            values = [data[f] for f in fields]

            sequence_name = f"seq_{self._table_name}_id"
            fields_with_id = ["id"] + fields
            placeholders_with_id = [f"nextval('{sequence_name}')"] + placeholders

            sql = f"""
            INSERT INTO {self._table_name} ({', '.join(fields_with_id)})
            VALUES ({', '.join(placeholders_with_id)})
            RETURNING id
            """
            result = self._connection.execute(sql, values).fetchone()
            self.id = result[0]
        else:
            # Update existing record
            set_clause = ", ".join([f"{f} = ${i+1}" for i, f in enumerate(data.keys())])
            values = list(data.values()) + [self.id]

            sql = f"""
            UPDATE {self._table_name}
            SET {set_clause}
            WHERE id = ${len(values)}
            """
            self._connection.execute(sql, values)

    @classmethod
    def get(cls: Type[T], id: int) -> Optional[T]:
        """Retrieve a model instance by ID."""
        if not cls._connection or not cls._table_name:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        result = cls._connection.execute(
            f"SELECT * FROM {cls._table_name} WHERE id = $1", [id]
        )
        columns = [desc[0] for desc in result.description]
        row = result.fetchone()

        if row:
            # Convert row to dict and handle special types
            data = dict(zip(columns, row))
            for key, value in data.items():
                if isinstance(value, str) and cls.__annotations__.get(key) in (
                    Dict,
                    Optional[Dict],
                ):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
            return cls(**data)
        return None

    @classmethod
    def list_all(cls: Type[T]) -> List[T]:
        """List all instances of the model."""
        if not cls._connection or not cls._table_name:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        result = cls._connection.execute(f"SELECT * FROM {cls._table_name}")
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        instances = []

        for row in rows:
            # Convert row to dict and handle special types
            data = dict(zip(columns, row))
            for key, value in data.items():
                if isinstance(value, str) and cls.__annotations__.get(key) in (
                    Dict,
                    Optional[Dict],
                ):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
            instances.append(cls(**data))

        return instances

    @classmethod
    def delete(cls, id: int) -> bool:
        """Delete a model instance by ID."""
        if not cls._connection or not cls._table_name:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        result = cls._connection.execute(
            f"DELETE FROM {cls._table_name} WHERE id = $1 RETURNING id", [id]
        ).fetchone()

        return result is not None

    @classmethod
    def list_all_filtered(cls: Type[T], field: str, value: Any) -> List[T]:
        """
        List all instances of the model filtered by a field value.

        Args:
            field (str): The field name to filter on
            value (Any): The value to filter by

        Returns:
            List[T]: List of model instances matching the filter

        Raises:
            RuntimeError: If database is not initialized
            ValueError: If field doesn't exist in the model
        """
        if not cls._connection or not cls._table_name:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        # Verify field exists in model
        if field not in cls.__annotations__:
            raise ValueError(f"Field '{field}' does not exist in model {cls.__name__}")

        # Handle special types for the filter value
        if isinstance(value, dict):
            value = json.dumps(value)
        elif isinstance(value, datetime):
            value = value.isoformat()

        result = cls._connection.execute(
            f"SELECT * FROM {cls._table_name} WHERE {field} = $1", [value]
        )
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        instances = []

        for row in rows:
            # Convert row to dict and handle special types
            data = dict(zip(columns, row))
            for key, value in data.items():
                if isinstance(value, str) and cls.__annotations__.get(key) in (
                    Dict,
                    Optional[Dict],
                ):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
            instances.append(cls(**data))

        return instances

    @classmethod
    def last(cls: Type[T]) -> Optional[T]:
        """Retrieve the last (highest ID) model instance."""
        if not cls._connection or not cls._table_name:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")

        result = cls._connection.execute(
            f"SELECT * FROM {cls._table_name} ORDER BY id DESC LIMIT 1"
        )
        columns = [desc[0] for desc in result.description]
        row = result.fetchone()

        if row:
            # Convert row to dict and handle special types
            data = dict(zip(columns, row))
            for key, value in data.items():
                if isinstance(value, str) and cls.__annotations__.get(key) in (
                    Dict,
                    Optional[Dict],
                ):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
            return cls(**data)
        return None
