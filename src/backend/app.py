from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from endpoints.routes import router as endpoint_router
from fastapi.staticfiles import StaticFiles

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
app.include_router(endpoint_router)
app.mount("/medical_images", StaticFiles(directory="medical_images"), name="medical_images")
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application"}


if __name__ == "__main__":
    app.run(debug=True)