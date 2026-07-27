"""
schemas.py — request/response models for the prediction API.

Pydantic validates every incoming request against these types automatically:
a bad payload gets a clear 422 error before it ever reaches the model.
"""

from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    """The raw order attributes needed to predict delivery time."""
    nb_items: int = Field(..., ge=1, description="Number of items in the order")
    n_sellers: int = Field(..., ge=1, description="Number of distinct sellers")
    n_categories: int = Field(..., ge=1, description="Number of distinct product categories")
    total_price: float = Field(..., ge=0, description="Total item price (BRL)")
    total_freight: float = Field(..., ge=0, description="Total freight value (BRL)")
    total_weight_g: float = Field(..., ge=0, description="Total order weight (grams)")
    avg_length_cm: float = Field(..., ge=0)
    avg_height_cm: float = Field(..., ge=0)
    avg_width_cm: float = Field(..., ge=0)
    purchase_hour: int = Field(..., ge=0, le=23, description="Hour of purchase (0-23)")
    purchase_dow: int = Field(..., ge=0, le=6, description="Day of week (0=Mon)")
    customer_state: str = Field(..., description="Two-letter state code, e.g. SP")
    customer_city: str = Field(..., description="Customer city name")
    customer_zip_code_prefix: int = Field(..., description="Customer zip prefix")

    model_config = {
        "json_schema_extra": {
            "example": {
                "nb_items": 2, "n_sellers": 1, "n_categories": 1,
                "total_price": 129.90, "total_freight": 18.30,
                "total_weight_g": 1500.0,
                "avg_length_cm": 30.0, "avg_height_cm": 10.0, "avg_width_cm": 20.0,
                "purchase_hour": 14, "purchase_dow": 2,
                "customer_state": "SP", "customer_city": "sao paulo",
                "customer_zip_code_prefix": 1310,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_delivery_days: float
