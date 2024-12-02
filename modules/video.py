import cv2
import torch
from torchvision import transforms, models
from PIL import Image

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the fine-tuned ResNet50 model
model = models.resnet50(pretrained=False)  # Not loading pre-trained weights here since you're using your own model
num_classes = len(emotion_labels)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the final fully connected layer
# model.fc = nn.Sequential(
#     nn.Dropout(0.3),  # Dropout layer with 30% probability
#     nn.Linear(model.fc.in_features, 7)  # Final classification layer for 7 classes
# ) # Add dropout layer and a new fully connected layer for expression_2 model
model.load_state_dict(torch.load("/Users/uktamnishonov/Desktop/AI_Surveillance_System/AI_Surveillance_System/models/expression_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define preprocessing transformations (matching your training setup)
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(  # Normalize with ImageNet mean and std
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]  # ImageNet std
    )
])

# Capture video from the laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect face using Haar cascades (grayscale frame used here for face detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Crop face from the original color frame (RGB)

        # Convert the cropped face from BGR (OpenCV default) to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Convert the face image (numpy array) to PIL Image before applying the transform
        pil_face = Image.fromarray(face_rgb)

        # Preprocess the face (resize, crop, normalize)
        processed_face = transform(pil_face).unsqueeze(0)  # Add batch dimension

        # Move the processed face to the appropriate device (CPU in this case)
        processed_face = processed_face.to(torch.device('cpu'))

        # Predict emotion
        with torch.no_grad():
            outputs = model(processed_face)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]

        # Draw a rectangle and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with detected faces and predicted emotions
    cv2.imshow("Facial Expression Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
