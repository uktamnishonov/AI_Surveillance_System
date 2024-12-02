import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Define emotion labels (match your model's classes)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load your Hugging Face model
@st.cache_resource  # Cache the model for better performance
def load_model():
    from torchvision import models
    import torch.nn as nn

    # Load the model architecture
    model = models.resnet50(pretrained=False)
    
    # Modify the final fully connected layer to match your classes
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Add dropout if used in training
        nn.Linear(model.fc.in_features, 7)  # 7 classes for facial expressions
    )
    
    # Load the weights
    model.load_state_dict(torch.load('/Users/uktamnishonov/Desktop/AI_Surveillance_System/AI_Surveillance_System/models/expression_model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("Real-Time Facial Expression Detection")
st.write("Activate your webcam to detect facial expressions in real time.")

# Initialize the video stream
run = st.checkbox('Activate Camera')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Unable to access the camera.")
        break

    # Convert the frame to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face using Haar cascades (you can optimize this later)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face and preprocess it
        face = rgb_frame[y:y+h, x:x+w]
        pil_face = Image.fromarray(face)
        processed_face = transform(pil_face).unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            outputs = model(processed_face)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]

        # Annotate the frame with the prediction
        cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(rgb_frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the annotated frame
    FRAME_WINDOW.image(rgb_frame)

cap.release()
