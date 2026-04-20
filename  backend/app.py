from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)
CORS(app)

# 🔹 define same model architecture
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)

# 🔹 load weights
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 🔹 Mapping for standard skin cancer classification (HAM10000 style)
# Indices: 0: Akiec, 1: Bcc, 2: Bkl, 3: Df, 4: Mel, 5: Nv, 6: Vasc
CLASS_MAPPING = {
    0: "Cancerous",
    1: "Cancerous",
    2: "Non-Cancerous",
    3: "Non-Cancerous",
    4: "Cancerous",
    5: "Non-Cancerous",
    6: "Non-Cancerous"
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, pred = torch.max(probabilities, 0)
            pred_idx = int(pred.item())
            
        status = CLASS_MAPPING.get(pred_idx, "Unknown")
        is_cancerous = status == "Cancerous"

        return jsonify({
            "prediction": pred_idx,
            "status": status,
            "is_cancerous": is_cancerous
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)