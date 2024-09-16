from pydantic import BaseModel, Field
from datetime import datetime, date
from decimal import Decimal


class TransactionModel(BaseModel):
    trans_date_trans_time: datetime = Field(..., alias="trans_date_trans_time")
    cc_num: str = Field(..., alias="cc_num")
    merchant: str = Field(..., alias="merchant")
    category: str = Field(..., alias="category")
    amt: Decimal = Field(..., alias="amt")
    first: str = Field(..., alias="first")
    last: str = Field(..., alias="last")
    gender: str = Field(..., alias="gender")
    street: str = Field(..., alias="street")
    city: str = Field(..., alias="city")
    state: str = Field(..., alias="state")
    zip: str = Field(..., alias="zip")
    lat: float = Field(..., alias="lat")
    long: float = Field(..., alias="long")
    city_pop: int = Field(..., alias="city_pop")
    job: str = Field(..., alias="job")
    dob: date = Field(..., alias="dob")
    trans_num: str = Field(..., alias="trans_num")
    unix_time: int = Field(..., alias="unix_time")
    merch_lat: float = Field(..., alias="merch_lat")
    merch_long: float = Field(..., alias="merch_long")
    is_fraud: bool = Field(..., alias="is_fraud")

    class Config:
        allow_population_by_field_name = True
