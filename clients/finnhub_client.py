import requests
import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"


class FinnhubClient:
    """Client for interacting with the Finnhub API."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Finnhub API key cannot be empty")
        self._token = api_key
        self._session = requests.Session()
        self._session.params = {"token": api_key}  # type: ignore[assignment]

    def _get(self, path: str, **params) -> Any:
        url = f"{FINNHUB_BASE}{path}"
        logger.info(f"[FINNHUB] GET {path} {params}")
        resp = self._session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Company profile: name, exchange, industry, market cap, logo, weburl."""
        return self._get("/stock/profile2", symbol=symbol)

    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Recent news articles for a company."""
        articles = self._get("/company-news", symbol=symbol, **{"from": from_date, "to": to_date})
        return [
            {
                "headline": a.get("headline"),
                "summary": a.get("summary"),
                "source": a.get("source"),
                "url": a.get("url"),
                "datetime": a.get("datetime"),
            }
            for a in (articles or [])
        ]

    def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """Key financial metrics: P/E, EPS, 52-week high/low, beta, etc."""
        data = self._get("/stock/metric", symbol=symbol, metric="all")
        return data.get("metric", {})
