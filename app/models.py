from typing import Literal
from pydantic import BaseModel, Field, model_validator


class UserProfile(BaseModel):
    """
    Validated investment profile for a user.

    This model is the single source of truth for a user's strategy.
    It is populated from a form (Streamlit, API, etc.) and validated
    before being used to build the LLM chain.

    Downstream uses:
      - profile.model_dump() → fills strategy_template via .partial()
      - Serialized to Postgres for persistence (Module 4)
    """

    # ── Core profile ────────────────────────────────────────────────
    horizon: str = Field(
        description="Investment time horizon, e.g. '6 months', '1-2 years', '10+ years'"
    )
    risk_level: Literal[
        "Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"
    ]
    strategy: Literal[
        "Growth", "Value", "Dividend Income", "Momentum", "Index / Passive", "Swing Trading"
    ]
    markets: Literal[
        "US (NYSE/NASDAQ)", "Israel (TASE)", "Europe", "Emerging Markets", "Global"
    ]
    currency: Literal["USD", "ILS", "EUR", "Multi-currency"]

    # ── Portfolio constraints ────────────────────────────────────────
    sectors: str = Field(
        description="Comma-separated sectors of interest, e.g. 'AI, SaaS, Energy'"
    )
    max_allocation_per_stock: int = Field(
        ge=1, le=100,
        description="Max % of portfolio allocated to a single stock"
    )
    min_positions: int = Field(ge=1, description="Minimum number of holdings")
    max_positions: int = Field(ge=1, description="Maximum number of holdings")
    rebalance_frequency: Literal[
        "Weekly", "Monthly", "Quarterly", "Annual", "No fixed schedule"
    ]
    benchmark: Literal[
        "S&P 500", "NASDAQ 100", "TA-125", "MSCI World", "Russell 2000"
    ]

    # ── Flags ────────────────────────────────────────────────────────
    esg_filter: bool = False
    leverage_allowed: bool = False

    # ── Optional free text ───────────────────────────────────────────
    custom_notes: str = ""

    # ── Cross-field validation ───────────────────────────────────────
    @model_validator(mode="after")
    def check_positions(self) -> "UserProfile":
        if self.min_positions > self.max_positions:
            raise ValueError(
                f"min_positions ({self.min_positions}) cannot exceed "
                f"max_positions ({self.max_positions})"
            )
        return self
