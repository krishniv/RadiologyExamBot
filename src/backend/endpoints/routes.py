import logging
from fastapi import APIRouter, HTTPException
from .Quizgen import quiz_generator  # Import the quiz generator instance
from .Medchat import Medchat

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/generate/{amount}")
async def generate_options(amount: int):
    """Generate quiz options for the requested amount of questions."""
    try:
        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0.")
        
        quiz_data = [quiz_generator.generate_question() for _ in range(amount)]
        return {"questions": quiz_data}
    
    except Exception as e:
        logger.error(f"Error generating options: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating quiz options.")

@router.post("/connectwallet")
async def connect_wallet(wallet_address: str):
    try:
        if not wallet_address:
            raise HTTPException(status_code=400, detail="Wallet address is required.")

        # Simulate saving or verifying the wallet address
        response = {
            "message": "Wallet connected successfully",
            "walletAddress": wallet_address,
        }
        return response
    
    except Exception as e:
        logger.error(f"Error connecting wallet: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while connecting the wallet.")

@router.post("/chat")
async def chat(user_message: str):
    try:
        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided.")

        response = Medchat.generate_response(user_message)

        return {"response": response}
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the chat.") 