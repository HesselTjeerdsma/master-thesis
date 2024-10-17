from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from datetime import datetime, date
from decimal import Decimal
import sys
import json
import pprint

sys.path.append("../")
from models.transaction import TransactionModel


def detect_fraud(transaction: TransactionModel, llm) -> dict:

    response_schemas = [
        ResponseSchema(
            name="fraud_risk", description="The level of fraud risk (Low/Medium/High)"
        ),
        ResponseSchema(
            name="reasons", description="Reasons for the fraud risk assessment"
        ),
        ResponseSchema(
            name="recommended_actions",
            description="Recommended actions based on the assessment",
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt = PromptTemplate(
        input_variables=["transaction_details"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
        template="""
        Analyze the following transaction details and determine if there's a potential for fraud. 
        Consider factors such as transaction amount, location, merchant details, and any unusual patterns.
        
        Transaction Details:
        {transaction_details}
        
        {format_instructions}
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    transaction_details = f"""
    Transaction Date/Time: {transaction.trans_date_trans_time}
    Amount: ${transaction.amt}
    Merchant: {transaction.merchant}
    Category: {transaction.category}
    Customer Name: {transaction.first} {transaction.last}
    Customer Location: {transaction.city}, {transaction.state}
    Transaction Location: Lat {transaction.lat}, Long {transaction.long}
    Merchant Location: Lat {transaction.merch_lat}, Long {transaction.merch_long}
    """

    try:
        result = chain.run(transaction_details=transaction_details)
        parsed_result = output_parser.parse(result)

        # Ensure the parsed result is JSON serializable
        json_result = json.loads(json.dumps(parsed_result))
        return json_result
    except Exception as e:
        # If there's an error, return a JSON object with an error message
        return {
            "error": str(e),
            "fraud_risk": "Unknown",
            "reasons": ["Error occurred during analysis"],
            "recommended_actions": ["Review transaction manually"],
        }
