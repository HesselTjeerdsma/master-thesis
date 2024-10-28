from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
from geopy.distance import geodesic
from typing import List, Dict
import sys
from dateutil.relativedelta import relativedelta

from dataclasses import dataclass
from datetime import datetime
from geopy.distance import geodesic
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

sys.path.append("../")
from models.transaction import TransactionModel
from pprint import pprint


@dataclass
class CardholderProfile:
    """Represents analyzed patterns of the cardholder"""

    home_location: tuple[float, float]  # (lat, long) of residential address
    common_merchants: List[str]  # Frequently visited merchants
    common_categories: List[str]  # Common spending categories
    typical_amounts: Dict[str, float]  # Typical amounts by category
    active_hours: List[int]  # Hours when cardholder typically transacts
    job: str  # Direct job title
    gender: str  # Gender (M/F)
    age: int  # Age derived from DOB
    usual_radius: float  # Typical transaction radius from home

    @classmethod
    def from_transaction(
        cls, transaction: TransactionModel, history: List[Dict]
    ) -> "CardholderProfile":
        """Create a cardholder profile from transaction and history"""
        # Calculate age
        dob = transaction.dob  # datetime.strptime(transaction.dob, "%Y-%m-%d")
        age = relativedelta(datetime.today(), dob).years
        # Home location from current transaction's address
        home_location = (float(transaction.lat), float(transaction.long))

        if not history:
            return cls(
                home_location=home_location,
                common_merchants=[transaction.merchant],
                common_categories=[transaction.category],
                typical_amounts={transaction.category: float(transaction.amt)},
                active_hours=[
                    datetime.strptime(
                        transaction.trans_date_trans_time, "%Y-%m-%d %H:%M:%S"
                    ).hour
                ],
                job=transaction.job,
                gender=transaction.gender,
                age=age,
                usual_radius=0.0,
            )

        # Analyze transaction history
        merchants = Counter([tx["merchant"] for tx in history])
        categories = Counter([tx["category"] for tx in history])

        # Calculate typical amounts by category
        amounts_by_category = {}
        for tx in history:
            cat = tx["category"]
            if cat not in amounts_by_category:
                amounts_by_category[cat] = []
            amounts_by_category[cat].append(tx["amount"])

        typical_amounts = {
            cat: sum(amounts) / len(amounts)
            for cat, amounts in amounts_by_category.items()
        }

        # Analyze transaction hours
        hours = [
            datetime.strptime(tx["timestamp"], "%Y-%m-%d %H:%M:%S").hour
            for tx in history
        ]

        # Calculate usual radius
        distances = [
            geodesic(
                home_location, (float(tx["merch_lat"]), float(tx["merch_long"]))
            ).miles
            for tx in history
        ]
        usual_radius = sum(distances) / len(distances)

        return cls(
            home_location=home_location,
            common_merchants=[m for m, _ in merchants.most_common(5)],
            common_categories=[c for c, _ in categories.most_common(5)],
            typical_amounts=typical_amounts,
            active_hours=list(set(hours)),
            job=transaction.job,
            gender=transaction.gender,
            age=age,
            usual_radius=usual_radius,
        )


def analyze_transaction_context(
    transaction: TransactionModel, history: List[Dict], profile: CardholderProfile
) -> Dict:
    """Analyze transaction in context of cardholder profile and history"""
    current_time = datetime.strptime(
        str(transaction.trans_date_trans_time), "%Y-%m-%d %H:%M:%S"
    )
    current_location = (float(transaction.merch_lat), float(transaction.merch_long))

    # Basic transaction analysis
    distance_from_home = geodesic(profile.home_location, current_location).miles

    # Category analysis
    category_typical_amount = profile.typical_amounts.get(transaction.category, 0)
    amount_deviation = (
        abs(float(transaction.amt) - category_typical_amount) / category_typical_amount
        if category_typical_amount > 0
        else 1.0
    )

    # Time pattern analysis
    hour = current_time.hour
    unusual_hour = hour not in profile.active_hours

    # Travel analysis if we have history
    travel_alert = None
    if history:
        last_tx = sorted(history, key=lambda x: x["timestamp"])[-1]
        last_time = datetime.strptime(last_tx["timestamp"], "%Y-%m-%d %H:%M:%S")
        last_location = (float(last_tx["merch_lat"]), float(last_tx["merch_long"]))

        if last_time < current_time:
            distance = geodesic(last_location, current_location).miles
            hours_diff = (current_time - last_time).total_seconds() / 3600
            speed = distance / hours_diff if hours_diff > 0 else 0

            if speed > 500:  # Faster than commercial flight
                travel_alert = f"Impossible travel speed: {speed:.1f} mph"

    return {
        "unusual_location": distance_from_home > (profile.usual_radius * 2),
        "unusual_amount": amount_deviation > 2.0,  # More than 2x typical amount
        "unusual_hour": unusual_hour,
        "unusual_merchant": transaction.merchant not in profile.common_merchants,
        "unusual_category": transaction.category not in profile.common_categories,
        "travel_alert": travel_alert,
        "distance_from_home": round(distance_from_home, 2),
        "amount_deviation": round(amount_deviation, 2),
        "demographic_context": {
            "age": profile.age,
            "gender": profile.gender,
            "job": profile.job,
        },
    }


def detect_fraud(
    transaction: TransactionModel, llm, transaction_history: List[Dict] = None
) -> dict:
    response_schemas = [
        ResponseSchema(
            name="risk_level", description="Risk level: LOW, MEDIUM, or HIGH"
        ),
        ResponseSchema(name="key_factors", description="List of key risk factors"),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    profile = CardholderProfile.from_transaction(transaction, transaction_history or [])
    risk_analysis = analyze_transaction_context(
        transaction, transaction_history or [], profile
    )

    prompt = PromptTemplate(
        input_variables=[
            "transaction",
            "profile",
            "risk_analysis",
            "history",
            "age",
            "gender",
            "job",
            "usual_radius",
        ],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
        template="""Analyze this transaction for fraud.

TRANSACTION: {transaction}
PROFILE: {profile}
RISK: {risk_analysis}
HISTORY: {history}

Consider:
1. Amount vs category norms
2. Location vs patterns
3. Time patterns
4. Merchant alignment
5. Age: {age}, Gender: {gender}, Job: {job}
6. Usual radius: {usual_radius:.1f} mi
7. Travel patterns

Return JSON with risk_level (LOW/MEDIUM/HIGH) and key_factors (list).

{format_instructions}""",
    )

    tx_details = f"${transaction.amt} at {transaction.merchant} ({transaction.category}), {transaction.city}, {transaction.state}, {risk_analysis['distance_from_home']}mi from home"

    profile_details = f"{profile.age}yo {profile.gender}, {profile.job}, radius: {profile.usual_radius:.1f}mi"

    risk_details = f"Location: {'High' if risk_analysis['unusual_location'] else 'Low'}, Amount: {'High' if risk_analysis['unusual_amount'] else 'Low'} ({risk_analysis['amount_deviation']}x), Time: {'High' if risk_analysis['unusual_hour'] else 'Low'}, Travel: {risk_analysis['travel_alert'] or 'None'}"

    history_details = (
        "None"
        if not transaction_history
        else ", ".join(
            [
                f"${tx['amount']} at {tx['merchant']}"
                for tx in sorted(
                    transaction_history, key=lambda x: x["timestamp"], reverse=True
                )[:2]
            ]
        )
    )

    try:
        demographic_data = risk_analysis["demographic_context"]
        formatted_prompt = prompt.format(
            transaction=tx_details,
            profile=profile_details,
            risk_analysis=risk_details,
            history=history_details,
            age=demographic_data["age"],
            gender=demographic_data["gender"],
            job=demographic_data["job"],
            usual_radius=profile.usual_radius,
        )

        chain = prompt | llm
        result = chain.invoke(
            {
                "transaction": tx_details,
                "profile": profile_details,
                "risk_analysis": risk_details,
                "history": history_details,
                "age": demographic_data["age"],
                "gender": demographic_data["gender"],
                "job": demographic_data["job"],
                "usual_radius": profile.usual_radius,
            }
        )

        return {"response": result, "prompt": formatted_prompt}

    except Exception as e:
        return {
            "prompt": formatted_prompt,
            "response": f"Error: {str(e)}",
        }
