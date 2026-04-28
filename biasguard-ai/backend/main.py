from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers import bias, simulate, recommend, demo, export, groq_insights

load_dotenv()

app = FastAPI(title="BiasGuard AI API")

# Configure CORS
origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bias.router)
app.include_router(simulate.router)
app.include_router(recommend.router)
app.include_router(demo.router)
app.include_router(export.router)
app.include_router(groq_insights.router)

@app.get("/")
async def root():
    return {"message": "Welcome to BiasGuard AI API"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "BiasGuard AI"}
