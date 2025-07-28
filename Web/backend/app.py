import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, Response
from torchvision import transforms, models
from PIL import Image
from flask_cors import CORS

# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LayerNorm(128)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        return (
            self.embedding_net(anchor),
            self.embedding_net(positive),
            self.embedding_net(negative),
        )

# Load full TripletNet
full_model = TripletNet(EmbeddingNet()).to(device)
full_model.load_state_dict(torch.load("model/triplet_face_model_2.pth", map_location=device))
full_model.eval()

# Extract embedding model
model = full_model.embedding_net
model.eval()

# ------------------- Transform -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------- Load Gallery -------------------
gallery_embeddings = []
gallery_labels = []
gallery_path = "gallery"

for person in os.listdir(gallery_path):
    person_dir = os.path.join(gallery_path, person)
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img_tensor).squeeze().cpu().numpy()
        gallery_embeddings.append(emb)
        gallery_labels.append(person)

# ------------------- Cosine Similarity -------------------
def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# ------------------- Webcam -------------------
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to PIL
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(img_tensor).squeeze().cpu().numpy()

        # Dự đoán người gần nhất
        sims = [cosine_sim(emb, ref) for ref in gallery_embeddings]
        if sims:
            best_idx = int(np.argmax(sims))
            label = gallery_labels[best_idx]
            score = sims[best_idx]
        else:
            label = "Unknown"
            score = 0.0

        # Hiển thị label
        cv2.putText(frame, f"{label} ({score:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Trả về frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ------------------- Routes -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------- Run -------------------
if __name__ == '__main__':
    app.run(debug=True)
    
app = Flask(__name__)
CORS(app)