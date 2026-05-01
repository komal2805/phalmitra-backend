"""
Phalmitra FastAPI Backend
Endpoint: POST /predict  — accepts a multipart image file, returns fruit prediction JSON.
"""

import io
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model_loader import ModelLoader

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Phalmitra API",
    description="Fruit recognition and quality assessment API using ResNet",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ──────────────────────────────────────────────────────
model_loader = ModelLoader()


@app.on_event("startup")
async def startup_event():
    logger.info("Loading model...")
    model_loader.load()
    logger.info("Model loaded successfully.")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "Phalmitra API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_loader.is_loaded()}


# ── Predict endpoint ───────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a fruit image via multipart upload and return prediction.

    Returns:
        {
            "fruit": "Apple",
            "confidence": 0.94,
            "quality": "Fresh"
        }
    """
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg",
                                  "image/webp", "image/bmp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG."
        )

    try:
        # Read image bytes
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # Open with PIL
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        result = model_loader.predict(image)

        logger.info(f"Prediction: {result}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
