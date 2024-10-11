from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from datetime import datetime, date
from decimal import Decimal
import sys

sys.path.append("../")
from models.transaction import TransactionModel

def detect_fraud(transaction: TransactionModel) -> dict:
    # Initialize the LocalLlama language model with GGUF file
    llm = LlamaCpp(
        model_path="/media/hessel/Media/lm-studio/bartowski/Phi-3.5-mini-instruct-GGUF/Phi-3.5-mini-instruct-Q8_0.gguf",
        temperature=0.1,
        max_tokens=2000,
        n_ctx=2048,
        n_batch=512,  # Increased for better GPU utilization
        n_gpu_layers=-1,  # Use all available GPU layers
        f16_kv=True,
        verbose=True,  # Set to False in production
        use_mlock=False,
        use_mmap=True
    )

    # Define the response schemas
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

    # Create the output parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Create a prompt template
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

    # Create an LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Prepare the transaction details
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

    # Run the chain
    result = chain.run(transaction_details=transaction_details)

    # Parse the result
    parsed_result = output_parser.parse(result)

    return parsed_result

# Create a test transaction
test_transaction = TransactionModel(
    trans_date_trans_time=datetime.now(),
    cc_num="4532015112830366",
    merchant="Tech Gadgets Online",
    category="Electronics",
    amt=Decimal("1999.99"),
    first="John",
    last="Doe",
    gender="M",
    street="123 Main St",
    city="New York",
    state="NY",
    zip="10001",
    lat=40.7128,
    long=-74.0060,
    city_pop=8336817,
    job="Software Engineer",
    dob=date(1985, 5, 15),
    trans_num="TR12345678",
    unix_time=int(datetime.now().timestamp()),
    merch_lat=34.0522,
    merch_long=-118.2437,
    is_fraud=False,  # We set this to False, but our function sho
)

# Uncomment the following lines to run the fraud detection
result = detect_fraud(test_transaction)
print(result)