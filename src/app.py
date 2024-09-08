from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from endpoints.generate_options import router as generate_options_router

app = FastAPI()

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; configure as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include the router from the generate_options module
app.include_router(generate_options_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application"}
