from pydantic import BaseModel, Field, field_validator
from datetime import date

class StockBase(BaseModel):
    """
    Base class for stock-related requests.
    """
    symbol: str = Field(
        ...,
        description="Stock ticker symbol as listed on the market (e.g., 'AAPL', 'MSFT').",
        min_length=1,
        max_length=10,
        example="AAPL"
    )


class StockInfoByDate(StockBase):
    """
    Represents a request for stock data within a specific date range.
    Inherits the 'symbol' field from StockBase.
    """
    from_date: date = Field(
        ...,
        description="Start date (inclusive) for fetching data, in YYYY-MM-DD format.",
        example="2024-01-01"
    )
    to_date: date = Field(
        ...,
        description="End date (inclusive) for fetching data, in YYYY-MM-DD format.",
        example="2024-12-31"
    )

    @field_validator("from_date", "to_date", mode="before")
    def validate_date_format(cls, value: str) -> date:
        try:
            return date.fromisoformat(value)
        except Exception:
            raise ValueError("Date must be in YYYY-MM-DD format (e.g., '2024-05-12').")

    @field_validator("to_date")
    def validate_date_order(cls, to_date_value, info):
        from_date_value = info.data.get("from_date")
        if from_date_value and to_date_value < from_date_value:
            raise ValueError("'to_date' must be greater than or equal to 'from_date'.")
        return to_date_value
