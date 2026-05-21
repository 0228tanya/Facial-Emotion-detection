import gradio as gr
from ultralytics import YOLO

# Load your model (ensure best.pt is in the same folder)
model = YOLO("best.pt")

def predict_emotion(image):
    # Perform inference
    results = model(image)
    # Return the annotated image
    return results[0].plot()

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_emotion, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Image(type="pil"),
    title="Facial Emotion Detection"
)

if __name__ == "__main__":
    demo.launch()