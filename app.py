import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os

# Set page configuration
st.set_page_config(page_title="Facial Emotion Detection", layout="centered")
st.title("🎬 Real-Time Facial Emotion Detection")
st.write("Upload a video file, and our trained YOLO model will detect facial emotions frame-by-frame.")

# 1. Load the trained model weights
@st.cache_resource  # Caches the model so it doesn't reload on every interaction
def load_model():
    # Looks for the weights file in the same directory
    return YOLO("facial_emotion_best.pt")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model. Make sure 'facial_emotion_best.pt' is in this folder. Details: {e}")
    st.stop()

# 2. Video Upload Component
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Open the video using OpenCV
    video_cap = cv2.VideoCapture(tfile.name)
    
    st.write("🔄 Processing video frames...")
    
    # Create an empty placeholder in Streamlit to update frames in real-time
    frame_placeholder = st.empty()
    
    # Stop button to interrupt processing
    stop_button = st.button("Stop Processing")

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret or stop_button:
            break
            
        # 3. Run YOLO inference on the current frame
        # stream=True optimizes memory management for videos
        results = model(frame, stream=True)
        
        # 4. Annotate the frame with bounding boxes and emotion labels
        for r in results:
            annotated_frame = r.plot()  # Returns the image with boxes/labels painted on
            
        # OpenCV reads BGR, but Streamlit displays RGB, so we convert it
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Update the placeholder with the newly processed frame
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
    video_cap.release()
    os.unlink(tfile.name)  # Clean up the temporary file
    st.success("Processing complete!")
    