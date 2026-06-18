# Dual Model Detection App

This project is a small computer vision web app that serves two local Ultralytics YOLO models:

- `fast_model.pt` for quick results on clear, simple images
- `thinking_model.pt` for harder cases, blurry images, or when maximum accuracy is requested

The app automatically chooses the better model based on the user's request and image size, then returns:

- the selected model name
- a short reason for the choice
- object detections with confidence scores
- an annotated preview image

## Features

- Web upload interface with a modern UI
- Automatic model selection based on request text and image difficulty
- JSON API for image analysis
- Annotated prediction output rendered in the browser

## Project Structure

- `app.py` - Flask backend and model selection logic
- `index.html` - Frontend upload page and results view
- `fast_model.pt` - speed-focused model
- `thinking_model.pt` - accuracy-focused model

## Requirements

- Python 3.13+
- Flask
- Pillow
- Ultralytics
- OpenCV headless

## Run Locally

```bash
python app.py
```

Then open the app in your browser at `http://127.0.0.1:5000`.

## API

### `POST /analyze`

Form fields:

- `image` - image file upload
- `request_text` - short text describing what kind of processing is needed
- `confidence` - optional confidence threshold

Response includes the selected model, the reason it was chosen, detections, and a base64 annotated image.

## Notes

The model choice is intentionally simple:

- if the request mentions blur, difficulty, or maximum quality, the app uses the thinking model
- if the request mentions speed, clarity, or simplicity, the app uses the fast model
- otherwise, smaller images default to the thinking model and larger clear images default to the fast model
