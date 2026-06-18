# Dual Model Detection App

This project is a Streamlit computer vision web app that serves two local Ultralytics YOLO models:

- `fast_model.pt` for quick results on clear, simple images
- `thinking_model.pt` for harder cases, blurry images, or when maximum accuracy is requested

The app automatically chooses the better model based on the user's request and image size, then returns:

- the selected model name
- a short reason for the choice
- object detections with confidence scores
- an annotated preview image

## Features

- Web upload interface with a modern UI built in Streamlit
- Automatic model selection based on request text and image difficulty
- Annotated prediction output rendered in the browser

## Project Structure

- `app.py` - Streamlit app and model selection logic
- `index.html` - Legacy frontend file, no longer used for deployment
- `fast_model.pt` - speed-focused model
- `thinking_model.pt` - accuracy-focused model

## Requirements

- Python 3.13+
- Streamlit
- Pillow
- Ultralytics
- OpenCV headless

## Run Locally

```bash
streamlit run app.py
```

Then open the app in your browser at the local Streamlit URL shown in the terminal.

## Streamlit Cloud Deployment

1. Push this repo to GitHub.
2. Open Streamlit Community Cloud and create a new app from this repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Notes

The model choice is intentionally simple:

- if the request mentions blur, difficulty, or maximum quality, the app uses the thinking model
- if the request mentions speed, clarity, or simplicity, the app uses the fast model
- otherwise, smaller images default to the thinking model and larger clear images default to the fast model
