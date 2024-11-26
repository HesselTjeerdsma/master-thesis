from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
from geopy.distance import geodesic
from typing import List, Dict
import sys
from dateutil.relativedelta import relativedelta
import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from collections import Counter

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from dateutil.relativedelta import relativedelta
from collections import Counter

sys.path.append("../")
from models.transaction import TransactionModel


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

        # remove current transcation from history
        # history = history[1:]

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

        # Fix: Move timestamp parsing inside the list comprehension
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
        # multiply by two as fair assumption
        usual_radius = (sum(distances) / len(distances)) * 2

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


def create_fraud_analysis_prompt(
    transaction: TransactionModel, profile: CardholderProfile, history: List[Dict]
) -> Tuple[PromptTemplate, Dict[str, Any]]:  # Note the return type change
    """
    Analyzes transaction context and creates a PromptTemplate for fraud detection.

    Returns:
        Tuple containing (prompt_template, input_values)
    """
    # Analyze transaction context
    current_location = (float(transaction.merch_lat), float(transaction.merch_long))
    distance_from_home = geodesic(profile.home_location, current_location).miles

    # Category analysis
    category_typical_amount = profile.typical_amounts.get(transaction.category, 0)
    amount_deviation = (
        abs(float(transaction.amt) - category_typical_amount) / category_typical_amount
        if category_typical_amount > 0
        else 1.0
    )

    # Time and pattern analysis
    hour = transaction.trans_date_trans_time.hour
    unusual_hour = hour not in profile.active_hours

    # Travel and history analysis
    travel_info = ""
    history_context = "No previous transaction history is available."
    last_time = None
    hours_diff = 0
    last_location = None

    if history:
        last_tx = history[0]
        last_time = datetime.strptime(last_tx["timestamp"], "%Y-%m-%d %H:%M:%S")
        last_location = (float(last_tx["merch_lat"]), float(last_tx["merch_long"]))

        if last_time < transaction.trans_date_trans_time:
            distance = geodesic(last_location, current_location).miles
            hours_diff = (
                transaction.trans_date_trans_time - last_time
            ).total_seconds() / 3600

            if hours_diff > 0:
                speed = distance / hours_diff
                if speed > 100:
                    travel_info = f" The transaction shows an unusually rapid change in location, indicating a travel speed of {speed:.1f} mph between transactions, which exceeds normal travel speeds."

            history_context = (
                f"Their last transaction was {last_time.strftime('%B %d at %I:%M %p')}, "
                f"approximately {hours_diff:.1f} hours ago, "
                f"which is {distance:.1f} miles from the current location."
            )

    # Create context strings for the narrative
    merchant_context = (
        "new to this customer"
        if transaction.merchant not in profile.common_merchants
        else "frequently visited by this customer"
    )
    category_context = (
        "unusual" if transaction.category not in profile.common_categories else "common"
    )
    time_context = (
        "outside their normal active hours"
        if unusual_hour
        else "within their typical active hours"
    )

    if category_context == "unusual":
        amount_context = "There is no transaction history for this category."
    else:
        amount_context = (
            f"The purchase amount is {'significantly higher' if amount_deviation > 2 else 'somewhat higher' if amount_deviation > 1.1 else 'typical'} "
            f"for this category of purchase."
        )
    # Location Analysis:
    # The transaction occurred {distance_from_home:.1f} miles from the customer's home location. On average all transactions for this customer happen in a range of {usual_radius:.1f} miles.{travel_info}

    # Create the prompt template using PromptTemplate
    template = """You are an expert fraud detection analyst within a financial institution's security system. Your role is to evaluate transactions for potential fraud, keeping in mind that your assessments will be reviewed by human analysts. You should flag any genuinely suspicious patterns while providing clear reasoning.


Transaction Context:
A {age}-year-old {gender_full} who works as a {job} has made a purchase of ${amount:.2f} at {merchant} ({category}) on {transaction_time}. This merchant is {merchant_context}, and this category of purchase is {category_context} for them. The transaction occurred {time_context}. {amount_context}


Customer Profile:
This customer typically shops at: {common_merchants}
Their usual purchase categories include: {common_categories}

Transaction History:
{history_context}

Please analyze this transaction for potential fraud indicators. Consider:
1. The location and travel patterns for age and job
2. Transaction amount and category
3. Timing and frequency
4. Any unusual patterns or deviations
5. If the transaction is typical for the age, job and gender

IMPORTANT: 
- Never repeat or reference the prompt instructions
- Never start with phrases like "Text for analysis:" or "Consider the transaction details"
- Always complete your full analysis in one clear statement
- Think step by step, but keep responses under 100 words total
- Only provide GENUINE or FRAUD as conclusion, do not use terms like UNCERTAIN or anything else.

Response Format:

For legitimate transactions:
[clear analysis in a single complete sentence]
CONCLUSION: GENUINE

For fraudulent transactions:
[clear analysis in a single complete sentence]
CONCLUSION: FRAUD

Example good response for fraud:
Transaction does not match the typical amount, timing, and location for this customer's profile.
CONCLUSION: FRAUD

Example good response for genuine transaction:
Transaction does not match the typical amount, timing, and location for this customer's profile.
CONCLUSION: GENUINE

Analyze the transcation and based on that analysis conclude if it is fraud or genuine.
"""

    # Create the prompt template
    prompt_template = PromptTemplate(
        template=template,
        input_variables=[
            "age",
            "gender",
            "job",
            "amount",
            "merchant",
            "category",
            "transaction_time",
            "merchant_context",
            "category_context",
            "time_context",
            "amount_context",
            "distance_from_home",
            "usual_radius",
            "travel_info",
            "common_merchants",
            "common_categories",
            "history_context",
        ],
    )

    gender_full = "female" if profile.gender == "F" else "male"

    # Create the input values dictionary
    input_values = {
        "amount": float(transaction.amt),
        "merchant": transaction.merchant,
        "category": transaction.category,
        "transaction_time": transaction.trans_date_trans_time.strftime(
            "%B %d at %I:%M %p"
        ),
        "merchant_context": merchant_context,
        "category_context": category_context,
        "time_context": time_context,
        "amount_context": amount_context,
        "distance_from_home": distance_from_home,
        "travel_info": travel_info,
        "common_merchants": ", ".join(profile.common_merchants)
        or "No established shopping patterns yet",
        "common_categories": ", ".join(profile.common_categories)
        or "No established category patterns yet",
        "history_context": history_context,
        "age": profile.age,
        "gender": profile.gender,
        "gender_full": gender_full,
        "job": profile.job,
        "usual_radius": profile.usual_radius,
    }

    return prompt_template, input_values


def detect_fraud(
    transaction: TransactionModel, llm, transaction_history: List[Dict] = None
) -> dict:
    """
    Detect fraud using LangChain and LLM
    """
    # Create the profile and analyze the transaction
    profile = CardholderProfile.from_transaction(transaction, transaction_history or [])

    # Create prompt template and input values
    prompt_template, input_values = create_fraud_analysis_prompt(
        transaction=transaction, profile=profile, history=transaction_history
    )

    if isinstance(llm, LlamaCpp):
        # Store original settings
        original_settings = {
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
            "top_p": llm.top_p,
            "top_k": llm.top_k,
        }

        # Apply strict settings
        llm.temperature = 0.6
        llm.max_tokens = 150

        # Create the chain properly using the prompt template
        chain = prompt_template | llm
        result = chain.invoke(input_values)

    formatted_prompt = prompt_template.format(**input_values)
    # this helps with crashes
    time.sleep(0.05)
    return {"response": result, "prompt": formatted_prompt}


def prepare_transaction_features(
    transaction, profile: Optional[CardholderProfile] = None
) -> pd.DataFrame:
    """Prepare transaction features for model input"""
    # Create a single row dataframe with the same features as training data
    df = pd.DataFrame(
        {
            "merchant": [transaction.merchant],
            "category": [transaction.category],
            "amt": [float(transaction.amt)],
            "gender": [transaction.gender],
            "lat": [float(transaction.lat)],
            "long": [float(transaction.long)],
            "city_pop": [transaction.city_pop],
            "job": [transaction.job],
            "unix_time": [int(transaction.trans_date_trans_time.timestamp())],
            "merch_lat": [float(transaction.merch_lat)],
            "merch_long": [float(transaction.merch_long)],
        }
    )

    # Add derived features if profile is available
    if profile:
        current_location = (float(transaction.merch_lat), float(transaction.merch_long))
        df["distance_from_home"] = [
            geodesic(profile.home_location, current_location).miles
        ]
        df["amount_typical"] = [
            (
                float(transaction.amt)
                / profile.typical_amounts.get(
                    transaction.category, float(transaction.amt)
                )
                if transaction.category in profile.typical_amounts
                else 1.0
            )
        ]
        df["is_common_merchant"] = [
            1 if transaction.merchant in profile.common_merchants else 0
        ]
        df["is_common_category"] = [
            1 if transaction.category in profile.common_categories else 0
        ]
        df["is_active_hour"] = [
            1 if transaction.trans_date_trans_time.hour in profile.active_hours else 0
        ]

    return df


def detect_fraud_conv(
    transaction,
    model_path: Optional[str] = None,
    transaction_history: List[Dict] = None,
) -> dict:
    """
    Detect fraud using conventional ML approach

    Args:
        transaction: Transaction object with attributes matching the training data
        model_path: Optional path to saved model file
        transaction_history: Optional list of previous transactions

    Returns:
        dict containing response and analysis
    """
    try:
        # Create profile from transaction history if available
        profile = None
        if transaction_history:
            profile = CardholderProfile.from_transaction(
                transaction, transaction_history
            )

        # Prepare features
        features_df = prepare_transaction_features(transaction, profile)

        # Encode categorical variables
        encoders = {}
        for col in ["merchant", "category", "gender", "job"]:
            encoders[col] = LabelEncoder()
            features_df[col] = encoders[col].fit_transform(features_df[col])

        # Create and train model if no saved model provided
        if model_path is None:
            model = SVC(probability=True)
            model.fit(features_df, [0])  # Fit with dummy target
        else:
            # Load model from path if provided
            # Implementation would depend on how model is saved
            pass

        # Get prediction probability
        fraud_prob = model.predict_proba(features_df)[0][1]

        # Generate analysis based on prediction
        if fraud_prob > 0.5:
            conclusion = "FRAUD"
            analysis = "Transaction shows unusual patterns in location, amount, or timing compared to typical behavior."
        else:
            conclusion = "GENUINE"
            analysis = (
                "Transaction matches expected patterns and typical customer behavior."
            )

        response = f"{analysis}\nCONCLUSION: {conclusion}"

        # Format output to match original function
        return {
            "response": response,
            "prompt": f"Analysis of transaction {transaction.trans_num} using conventional ML model",
        }

    except Exception as e:
        return {
            "response": f"Error analyzing transaction: {str(e)}\nCONCLUSION: ERROR",
            "prompt": "Error occurred during analysis",
        }
