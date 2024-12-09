from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import cv2
from fastapi import FastAPI, Request, File, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from ultralytics import YOLO
import io
from pathlib import Path
import os

# Initialize FastAPI app
app = FastAPI()

# Mount static directory for serving processed images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model
# model_path = "assignment\best.pt"  # Update your path
# model_path = os.path.join("assignment", "best.pt")
model_path = 'models/best.pt'
model = YOLO(model_path)

# Templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Create static and templates directories if not exist
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page with upload form and optional result."""
    return templates.TemplateResponse("home.html", {"request": request, "image_url": None})



@app.post("/", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    """Process uploaded image and display results."""
    # Load image
    file_content = await file.read()
    image = Image.open(io.BytesIO(file_content))
    orig_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform detection
    results = model(np.array(image))

    # Retrieve class names
    class_names = model.names

    # Draw bounding boxes and labels
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = float(box.conf[0])  # Confidence score
        class_id = int(box.cls[0])  # Class index
        class_name = class_names[class_id]  # Class name
        label = f"{class_name} {confidence:.2f}"

        # Draw the bounding box
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        label_y = max(y1, label_size[1] + 10)
        cv2.rectangle(orig_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Save the processed image
    result_path = f"static/result_{file.filename}"
    cv2.imwrite(result_path, orig_img)

    # Render the results on the same page
    return templates.TemplateResponse(
        "home.html", {"request": request, "image_url": f"/static/result_{file.filename}"}
    )





