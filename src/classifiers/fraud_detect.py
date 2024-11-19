from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
from geopy.distance import geodesic
from typing import List, Dict
import sys
from dateutil.relativedelta import relativedelta
import json
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from collections import Counter
from llama_cpp import LlamaGrammar

sys.path.append("../")
from models.transaction import TransactionModel

# Update the grammar to be absolutely strict about the output
FRAUD_DETECTION_GRAMMAR_STRING = r"""
root   ::= json
json   ::= "{" ws risk_level ws "," ws key_factors ws "}"
risk_level ::= "\"risk_level\"" ws ":" ws ("\"LOW\"" | "\"MEDIUM\"" | "\"HIGH\"")
key_factors ::= "\"key_factors\"" ws ":" ws "[" factors "]"
factors ::= "" | factor_item | factor_item ("," ws factor_item)*
factor_item ::= "\"" [^""]+ "\""
ws     ::= [ \t\n]*
"""


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
        dob = transaction.dob
        age = relativedelta(datetime.today(), dob).years
        # Home location from current transaction's address
        home_location = (float(transaction.lat), float(transaction.long))

        if not history:
            return cls(
                home_location=home_location,
                common_merchants=[transaction.merchant],
                common_categories=[transaction.category],
                typical_amounts={transaction.category: float(transaction.amt)},
                active_hours=[transaction.trans_date_trans_time.hour],
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
        timestamp = datetime.strptime(
            tx["timestamp"], "%Y-%m-%d %H:%M:%S"
        )  # Analyze transaction hours
        hours = [timestamp.hour for tx in history]

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
    current_time = transaction.trans_date_trans_time
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


def create_risk_prompt(
    tx_details: str,
    profile_details: str,
    risk_details: str,
    history_details: str,
    demographic_data: Dict[str, Any],
    usual_radius: float,
) -> tuple[PromptTemplate, dict]:
    """Creates the prompt template and its input values"""
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
        template="""You are an expert in detecting fraud, with expertise in financial transaction analysis. 
        Your role is to identify suspicious patterns and anomalies in transactions. 
        You are part of a multi-layer fraud detection system where your flags will be reviewed by human analysts. 
        Since false negatives (missing fraud) are more costly than false positives (flagging legitimate transactions), err on the side of caution when flagging suspicious activity.

Input Data:
TRANSACTION: {transaction}
PROFILE: {profile}
HISTORY: {history}
AGE: {age}
GENDER: {gender}
JOB: {job}
USUAL RADIUS: {usual_radius:.1f} mi

{risk_analysis}
""",
    )

    input_values = {
        "transaction": tx_details,
        "profile": profile_details,
        "risk_analysis": risk_details,
        "history": history_details,
        "age": demographic_data["age"],
        "gender": demographic_data["gender"],
        "job": demographic_data["job"],
        "usual_radius": usual_radius,
    }

    return prompt, input_values


def detect_fraud(
    transaction: TransactionModel, llm, transaction_history: List[Dict] = None
) -> dict:
    """
    Detect fraud using GBNF grammar for structured output
    """
    # First create the profile and analyze the transaction
    profile = CardholderProfile.from_transaction(transaction, transaction_history or [])
    risk_analysis = analyze_transaction_context(
        transaction, transaction_history or [], profile
    )

    # Prepare all the details
    tx_details = f"${transaction.amt} at {transaction.merchant} ({transaction.category}), {transaction.city}, {transaction.state}, {risk_analysis['distance_from_home']}mi from home"
    profile_details = f"{profile.age}yo {profile.gender}, {profile.job}, radius: {profile.usual_radius:.1f}mi"
    risk_details = f"""The transaction location is {'unusually far from typical patterns' if risk_analysis['unusual_location'] else 'within normal travel range'}
The transaction amount is {'significantly higher than usual' if risk_analysis['unusual_amount'] else 'consistent with past spending'} ({risk_analysis['amount_deviation']:.1f}x typical)
The timing of this transaction {'falls outside normal hours' if risk_analysis['unusual_hour'] else 'matches typical patterns'}
{risk_analysis['travel_alert'] if risk_analysis['travel_alert'] else 'No concerning travel patterns detected'}"""
    history_details = (
        "None"
        if not transaction_history
        else ", ".join(
            [
                f"${tx['amount']} at {tx['merchant']}"
                for tx in sorted(
                    transaction_history, key=lambda x: x["timestamp"], reverse=True
                )[1:10]
            ]
        )
    )

    try:
        # Create prompt and input values
        prompt, input_values = create_risk_prompt(
            tx_details,
            profile_details,
            risk_details,
            history_details,
            risk_analysis["demographic_context"],
            profile.usual_radius,
        )

        formatted_prompt = prompt.format(**input_values)

        # Create LlamaGrammar object from the grammar string
        grammar = LlamaGrammar.from_string(FRAUD_DETECTION_GRAMMAR_STRING)

        # Set the grammar on the LLM
        if isinstance(llm, LlamaCpp):
            # Store original settings
            original_settings = {
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
                "top_p": llm.top_p,
                "top_k": llm.top_k,
            }

            # Apply strict settings
            llm.temperature = 0.8
            llm.max_tokens = 500  # Limit output length
            llm.client.grammar = grammar

            try:
                chain = prompt | llm
                result = chain.invoke(input_values)
            finally:
                # Restore original settings
                llm.temperature = original_settings["temperature"]
                llm.max_tokens = original_settings["max_tokens"]
                llm.top_p = original_settings["top_p"]
                llm.top_k = original_settings["top_k"]

        return {"response": result, "prompt": formatted_prompt}

    except Exception as e:
        return {
            "prompt": formatted_prompt,
            "response": f"Error: {str(e)}",
        }
