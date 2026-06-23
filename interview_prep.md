# Skin Cancer Detection AI: Comprehensive Interview Preparation Guide

This guide is designed to prepare you to explain, defend, and walk through the **Skin Cancer Detection AI** project during your upcoming interview. It details everything from high-level architecture down to line-by-line model inference details, trade-offs, and typical interviewer "trap" questions.

---

## 1. Project Overview

### Project Name
* **Skin Cancer Detection AI** (often referred to as **SkinCare AI** or **OE Mini Project**).

### Problem It Solves
Dermatological health checking is often bottlenecked by expensive consultations and long waitlists. Early detection of malignant skin lesions (like Melanoma or Basal Cell Carcinoma) dramatically improves survival rates. This project provides an accessible, instant, and high-accuracy preliminary analysis tool that classifies uploaded skin lesion images into **Cancerous** or **Non-Cancerous** categories using a state-of-the-art Deep Learning model.

### Target Users
* **General Public / Patients**: Seeking immediate, preliminary screening of a suspicious mole or skin lesion before consulting a dermatologist.
* **Primary Care Physicians**: Requiring a rapid diagnostic support tool to triage cases.

### Key Features
1. **High-Aesthetic UI**: A modern Glassmorphism-style single-page interface with smooth micro-animations, drag/upload dropzone, and responsive feedback.
2. **Instant Neural Classification**: Uploads skin photographs and evaluates them using a fine-tuned EfficientNet-B0 model trained on the standard HAM10000 dataset.
3. **Simplified Result Mapping**: Abstracts complex medical classification terminology (e.g., *Actinic keratoses*, *Melanoma*) into actionable, binary categories ("Cancerous" vs "Non-Cancerous") for patient clarity.
4. **Interactive Image Preview & Diagnostics**: Allows immediate removal/change of photos with visual validation.

### Business Value
* **Early Triage & Intervention**: Reduces burden on clinical pipelines by screening out clearly benign lesions (e.g., common moles, benign keratosis) while alerting high-risk users to seek urgent care.
* **Privacy-First Design**: The lightweight, stateless API does not persist images in a database, ensuring immediate compliance with medical data privacy regulations (HIPAA/GDPR).

---

## 2. Architecture Analysis

The system is designed as a decoupled, client-server web application separating the presentation logic from the computational workload of PyTorch deep learning.

### Overall System Architecture Diagram
```
                     +---------------------------------------+
                     |         Frontend (React + Vite)       |
                     |  - Renders Glassmorphism UI           |
                     |  - Handles image selection & preview  |
                     |  - Serializes image as Multipart Form  |
                     +-------------------+-------------------+
                                         |
                                         | HTTP POST (Multipart/Form-Data)
                                         | Target: /predict (Port 5001)
                                         v
                     +-------------------+-------------------+
                     |          Backend (Flask API)          |
                     |  - CORS Enabled for client origins    |
                     |  - Receives uploaded image stream     |
                     |  - Performs normalization transforms  |
                     |  - EfficientNet-B0 PyTorch Inference  |
                     +-------------------+-------------------+
                                         |
                                         | Evaluates weights from
                                         v
                             [ model.pth (PyTorch Weights) ]
```

### Frontend Technologies
* **Vite + React (v19)**: Selected for lightning-fast hot module replacement (HMR), lightweight bundle size, and modular structure.
* **Axios**: Handles asynchronous HTTP requests to transfer multipart form data.
* **Vanilla CSS (Outfit Google Font)**: Leverages custom CSS properties, backdrop filters (`blur(16px)`), CSS transitions, animations, and custom radial-gradient backgrounds for a premium aesthetic.

### Backend Technologies
* **Python v3 + Flask**: Used to build a lightweight, routing-efficient REST API.
* **PyTorch (torch, torchvision)**: The core machine learning library handling model instantiation, weight loading, image tensor conversion, and forward-pass inference.
* **PIL (Pillow)**: Utilized to handle image stream ingestion and RGB conversion.
* **Flask-CORS**: Permits cross-origin requests from the Vite frontend development server.

### Database Design
* **Stateless Operation (No Database)**: The application processes images purely in-memory. The image is uploaded, parsed, run through the model, and discarded after returning the prediction. 
* *Rationale*: Enhances security and privacy compliance (HIPAA), prevents server-side storage bloat, and optimizes response latency.

### APIs and Integrations
* **Predict Endpoint (`POST /predict`)**:
  * **Payload**: Multipart form data with key `file` containing the image.
  * **Response JSON**:
    ```json
    {
      "prediction": 4,
      "status": "Cancerous",
      "is_cancerous": true
    }
    ```

### Folder Structure Explanation
```
oe-mini-project/
│
├── OE_Fixed.ipynb          # Jupyter Notebook used to download, preprocess, train, fine-tune, and evaluate the model.
├──  backend/               # Flask Application (Note the leading space in directory name)
│   ├── app.py              # Flask server and prediction API route
│   └── model.pth           # Serialized state dictionary of the trained PyTorch model (16.3 MB)
│
├── frontend/               # Vite-React frontend project
│   ├── index.html          # HTML entry point (Mounts main.jsx)
│   ├── package.json        # Node dependency manifest
│   ├── vite.config.js      # Vite compilation configuration
│   └── src/
│       ├── main.jsx        # App mounting entry point
│       ├── App.jsx         # Core UI view & request handlers
│       ├── index.css       # Complete custom glassmorphism design system & variables
│       └── App.css         # Component specific styling overrides
│
└── test_samples/           # Test skin lesion images to verify model accuracy manually
```

---

## 3. Technical Deep Dive

### Major Modules & Code Flow

#### 1. Model Training & Fine-Tuning Pipeline ([OE_Fixed.ipynb](file:///Users/larenpinto/Desktop/oe-mini-project/OE_Fixed.ipynb))
* **Dataset**: HAM10000 dataset (downloaded via `kagglehub` from `tsaideepak/skin-cancer`).
* **Imbalance Treatment**: 
  * Class weights are calculated as $W_i = \text{total} / \text{count}_i$ using `collections.Counter`.
  * Weighted class loss: Passed directly to `nn.CrossEntropyLoss(weight=weights)` to penalize minority-class errors heavily.
* **Training Strategy (Two-Stage Fine-Tuning)**:
  * **Stage 1**: Freezes the convolutional backbone (`pretrained=True`) and trains only the custom classifier head (Linear layer mapping from features to 7 classes) for 5 epochs with a learning rate of $3 \times 10^{-4}$.
  * **Stage 2**: Unfreezes all layers and fine-tunes the entire network for 20 epochs with a lower learning rate of $1 \times 10^{-4}$ and a Cosine Annealing Learning Rate Scheduler (`CosineAnnealingLR`) to converge smoothly.

#### 2. Inference API Server ([app.py](file:///Users/larenpinto/Desktop/oe-mini-project/%20backend/app.py))
* **Model Loading**: Re-instantiates `models.efficientnet_b0()`, alters the classifier output features count to 7, and loads weights via `torch.load("model.pth", map_location="cpu")`. Using `map_location="cpu"` is crucial, ensuring the model runs perfectly in production/local CPU environments without relying on CUDA.
* **Transform Pipeline**: Preprocesses incoming images to match the exact distribution the model was trained on:
  1. `Resize((224, 224))` - Scaled to the input dimensions expected by EfficientNet-B0.
  2. `ToTensor()` - Converts the image into a PyTorch float tensor scaled to $[0.0, 1.0]$.
  3. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` - Standardizes pixel values using ImageNet mean/standard deviation.
* **Inference Block**:
  * Wraps calculations with `torch.no_grad()` to turn off the autograd engine, reducing memory footprint and accelerating inference.
  * Applies `torch.nn.functional.softmax()` to output logits to convert raw outputs to probabilities.
  * extracts the index with highest probability (`torch.max(probabilities, 0)`).

#### 3. Client UI ([App.jsx](file:///Users/larenpinto/Desktop/oe-mini-project/frontend/src/App.jsx))
* **Image Staging**: When a user selects a file, `URL.createObjectURL(selectedFile)` constructs a temporary local URL to render a high-speed preview without waiting for upload.
* **Request Assembly**: Packages the file in standard HTTP `FormData`.
* **State Updates**: Updates `result`, `loading`, and `preview` states, dynamically showing/hiding indicators, changing button hover states, and triggering the `.result-container` animation.

### Data Flow Through the Application

```
[ User selects Image ] --> (Generate local ObjectURL -> Show Preview)
                                   |
                           (Click "Analysis")
                                   |
                                   v
             [ Axios POST FormData with image file ]
                                   |
                                   v
                      [ Flask Server /predict ]
                                   |
                        (Image converted to RGB)
                                   |
                     (Resized & ImageNet Normalized)
                                   |
                        (EfficientNet Inference)
                                   |
                      (Argmax Class ID Extracted)
                                   |
                  (Mapped: Class ID -> Binary Label)
                                   |
                                   v
               [ JSON Response returned to React client ]
                                   |
                                   v
             [ State Update -> Triggers CSS transition ]
                                   |
                                   v
              [ Result Banner Displays "Cancerous" (Red) ]
                           or "Non-Cancerous" (Green)
```

---

## 4. Database Documentation

As identified in the Architecture section, this project is designed as a **stateless application** that processes uploads on-the-fly and does not store user data or images in a database.

### What if an interviewer asks: "How would you design a database for this?"
If requested to design a database schema for an enterprise version, here is the relational architecture you should propose:

#### Proposed Schema Design
```
                       +------------------+
                       |      users       |
                       +------------------+
                       | PK | id          |
                       |    | email       |
                       |    | password    |
                       +---------+--------+
                                 | 1
                                 |
                                 | 1..*
                       +---------v--------+
                       |    scans_log     |
                       +------------------+
                       | PK | scan_id     |
                       | FK | user_id     |
                       |    | img_s3_url  |
                       |    | pred_class  |
                       |    | confidence  |
                       |    | is_cancerous|
                       |    | created_at  |
                       +------------------+
```

#### CRUD Operations in this Extended System:
* **Create**: `POST /scans` would upload the raw image to an object store (e.g. AWS S3), run the Flask prediction model, and write the metadata (predicted class, confidence score, user id) to the `scans_log` table.
* **Read**: `GET /scans` retrieves previous analyses for a logged-in user to show their history or track lesion evolution.
* **Update**: `PATCH /scans/:id` allows a physician or dermatologist to override the AI label with a biopsy-verified classification (e.g. updating the "ground truth" for future retrainings).
* **Delete**: `DELETE /scans/:id` purges records at the user's request (necessary for HIPAA compliance and the "Right to be Forgotten" under GDPR).

---

## 5. Interview Questions and Answers

### Beginner Level (1-10)

#### 1. What is the core stack of this application?
* **Answer**: The application uses React with Vite on the frontend and Python with Flask and PyTorch on the backend. No database is used to keep the app lightweight and secure.

#### 2. What dataset was the model trained on?
* **Answer**: The model was trained on the HAM10000 dataset ("Human Against Machine with 10,000 training images"), which is a widely-used benchmark dataset for skin cancer detection containing dermatoscopic images.

#### 3. How does the frontend communicate with the backend?
* **Answer**: The frontend uses **Axios** to send an HTTP `POST` request to `http://127.0.0.1:5001/predict`. The payload is wrapped in a `FormData` object containing the raw image file.

#### 4. Why are the frontend and backend running on different ports?
* **Answer**: To isolate concerns. The Vite development server runs on its default port, while the Flask backend is configured to listen on port `5001`. This requires enabling **CORS (Cross-Origin Resource Sharing)** on the backend to allow cross-origin requests.

#### 5. What is the role of `CORS(app)` in `app.py`?
* **Answer**: By default, web browsers block frontend code from making requests to an API hosted on a different origin (port/domain) to prevent cross-site attacks. `CORS(app)` enables the Cross-Origin Resource Sharing headers, allowing the React frontend (running on a different port) to access Flask's endpoints.

#### 6. What does `transforms.Resize((224, 224))` do in the pipeline?
* **Answer**: Neural Networks require static-shaped inputs. EfficientNet-B0 was designed and pretrained on $224 \times 224$ pixel images. This transform resizes any uploaded user image, regardless of aspect ratio, to exactly $224 \times 224$ pixels before passing it to the model.

#### 7. Why is the image converted using `.convert("RGB")` in the backend?
* **Answer**: Users can upload images in various formats (e.g., PNG, RGBA, Grayscale). PNGs often have 4 channels (RGBA with transparency). Converting to "RGB" strips the alpha channel, ensuring the model always receives a standard 3-channel input tensor.

#### 8. What is the purpose of `torch.no_grad()` during inference?
* **Answer**: During training, PyTorch constructs a dynamic computation graph to calculate gradients for backpropagation. During inference, we do not update weights. Wrapping code in `torch.no_grad()` prevents PyTorch from tracking gradients, which cuts down memory consumption by up to 50% and speeds up inference execution.

#### 9. Why is the prediction value converted to binary "Cancerous" / "Non-Cancerous" labels in the response?
* **Answer**: The HAM10000 dataset classifies lesions into 7 fine-grained medical terms. These terms are confusing and alarming to general users. The backend aggregates them into a user-friendly binary classification: malignant classes (Akiec, Bcc, Mel) are labeled "Cancerous", while benign ones (Bkl, Df, Nv, Vasc) are labeled "Non-Cancerous".

#### 10. What is Vite and why did you use it instead of Create React App (CRA)?
* **Answer**: Vite is a modern frontend build tool that leverages native ES modules to compile code on-demand in development, avoiding the slow bundling process of Webpack. It compiles changes instantly (Fast HMR) and produces optimized, lightweight assets during builds.

---

### Intermediate Level (11-20)

#### 11. Explain the architecture of EfficientNet-B0. Why was it chosen over ResNet50?
* **Answer**: EfficientNet uses a technique called **compound scaling** to scale network depth, width, and resolution uniformly using a fixed compound coefficient. Traditional networks scale only one dimension. EfficientNet-B0 has 5.3 million parameters (compared to ResNet50's ~25 million parameters) but achieves comparable or superior accuracy on ImageNet. This makes it ideal for deployment on CPU-based servers or edge systems, offering low latency and low memory footprint.

#### 12. What does `unsqueeze(0)` do to the image tensor in `app.py`?
* **Answer**: Deep learning models expect batches of inputs, with a shape of `[batch_size, channels, height, width]`. The transform pipeline outputs a single image tensor of shape `[3, 224, 224]`. `unsqueeze(0)` adds a batch dimension at index 0, turning it into `[1, 3, 224, 224]`, which is compatible with the model.

#### 13. How did you handle class imbalance in the training data?
* **Answer**: Class imbalance is a major problem in HAM10000 (Melanocytic nevi make up ~67% of the dataset). In the notebook [OE_Fixed.ipynb](file:///Users/larenpinto/Desktop/oe-mini-project/OE_Fixed.ipynb), I handled this by calculating inverse frequency weights for each class and passing them to the Cross Entropy Loss function: `nn.CrossEntropyLoss(weight=weights)`. This scales the loss so that misclassifications of minority classes (like Dermatofibroma) penalize the model more than misclassifications of dominant classes.

#### 14. What are the two stages of fine-tuning implemented in the training code?
* **Answer**: 
  1. **Stage 1 (Feature Extraction)**: The convolutional backbone weights are frozen. Only the newly initialized classifier head is trained for 5 epochs. This prevents the large gradients from the randomly initialized head from destroying the pretrained features in the backbone.
  2. **Stage 2 (Full Fine-Tuning)**: All layers are unfrozen, and the entire network is trained at a much lower learning rate ($1 \times 10^{-4}$) for 20 epochs with a Cosine Annealing scheduler. This allows the backbone to adjust its weights subtly to dermatoscopic image features.

#### 15. Explain ImageNet normalization. Why are the exact mean and std values (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`) used?
* **Answer**: These values represent the channel-wise mean and standard deviations of the entire ImageNet dataset. Since we are using an EfficientNet model pretrained on ImageNet, we must normalize our input skin images to have the exact same distribution. Normalizing centers the data and reduces internal covariate shift, ensuring the pre-trained weights extract features correctly.

#### 16. What is the mathematical operation behind `torch.nn.functional.softmax(output, dim=1)`?
* **Answer**: Softmax converts the model's raw output values (logits) into a probability distribution. It exponentiates each logit and divides it by the sum of all exponentiated logits:
  $$P(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
  This forces the outputs to lie in the range $[0, 1]$ and sum up to $1.0$, which allows us to interpret them as confidence probabilities.

#### 17. How does the frontend handle the loading state during inference?
* **Answer**: It uses a Boolean state `loading`. When the user clicks the prediction button, `loading` is set to `true`. This disables the button to prevent double-submissions and replaces the button text with a CSS animated loading spinner (`.loading-spinner`). In the `finally` block of the async Axios call, `loading` is reset to `false`.

#### 18. What is `URL.createObjectURL()` and why is it used?
* **Answer**: `URL.createObjectURL(file)` creates a temporary, DOM-string URL representing the selected file in local browser memory. Using this local URL, the frontend can instantly display the selected image as a preview without sending it over the network to the server first, improving responsiveness.

#### 19. Why does `app.py` load the model with `map_location="cpu"`?
* **Answer**: The training notebook was executed on a GPU (e.g. Google Colab T4 GPU), saving model weights as CUDA tensors. When loading these weights on a production or local server that may only have a CPU, PyTorch will error out trying to look for a GPU. `map_location="cpu"` intercepts the load and maps all CUDA tensors to CPU memory automatically.

#### 20. What is the activation function of the final layer in EfficientNet-B0?
* **Answer**: The final classifier layer is a linear layer (`nn.Linear`) that outputs raw logits. No activation is applied inside the model's definition itself. The activation (Softmax) is applied externally during inference in `app.py` to extract probabilities, and `CrossEntropyLoss` (which combines LogSoftmax and NLLLoss) is used during training.

---

### Advanced Level (21-30)

#### 21. How would you improve the security of the current Flask endpoint?
* **Answer**: 
  1. **Strict File Validation**: Currently, it checks `Image.open(file)`. A malicious user could upload a script renamed to `.jpg`. I would implement a file extension whitelist and check the mime-type.
  2. **Rate Limiting**: Integrate `Flask-Limiter` to prevent Denial of Service (DoS) attacks on the computation-heavy inference endpoint.
  3. **Turn off Debug Mode**: Currently, `debug=True` is active. In production, this must be `False` as Flask's interactive debugger allows arbitrary code execution if an unhandled exception occurs.

#### 22. What is Cosine Annealing learning rate scheduling, and why was it chosen?
* **Answer**: Cosine Annealing decreases the learning rate following a cosine curve. It starts high to rapidly cross local minima, and then decreases slowly before dropping sharply near the end of training. It allows the model to settle into deep, stable minima, preventing oscillation around the optimal weights and improving generalization.

#### 23. If you wanted to run this model directly in the browser, how would you do it?
* **Answer**: I would export the trained PyTorch model to the **ONNX (Open Neural Network Exchange)** format and run it on the frontend using **ONNX Runtime Web**. This would eliminate backend hosting costs and latency completely, while protecting patient privacy by executing inference entirely on the client's device.

#### 24. Explain the HAM10000 classes. Which indices map to what diseases?
* **Answer**:
  * **0: Akiec**: Actinic Keratoses / Bowen's Disease (precancerous/malignant)
  * **1: Bcc**: Basal Cell Carcinoma (malignant)
  * **2: Bkl**: Benign Keratosis (benign)
  * **3: Df**: Dermatofibroma (benign)
  * **4: Mel**: Melanoma (highly malignant)
  * **5: Nv**: Melanocytic Nevi / Moles (benign)
  * **6: Vasc**: Vascular lesions (benign)

#### 25. Why did you use AdamW optimizer instead of standard Adam or SGD?
* **Answer**: AdamW modifies standard Adam by decoupling weight decay from the gradient updates. In standard Adam, weight decay is added to the gradient, which distorts the moving averages of gradients. Decoupling weight decay in AdamW ensures correct regularization, resulting in better generalization and training stability, especially when fine-tuning deep architectures like EfficientNet.

#### 26. How would you optimize the model latency on a CPU-bound server?
* **Answer**:
  1. **Quantization**: Convert the model weights from float32 (32-bit floating point) to int8 (8-bit integer) using PyTorch's Post-Training Static Quantization. This reduces model size by 75% and speeds up inference on CPUs with vector extensions.
  2. **TorchScript / Compilation**: Compile the model using `torch.compile()` or JIT to optimize operators.
  3. **Inference engine**: Deploy the model using a dedicated CPU inference engine like OpenVINO or ONNX Runtime.

#### 27. What metric would you prioritize during evaluation: Precision or Recall?
* **Answer**: In medical diagnostics, **Recall (Sensitivity)** is the top priority. Recall measures the percentage of actual cancerous lesions that were correctly identified. A false positive (high recall, lower precision) means a patient gets a biopsy on a benign mole, which is manageable. A false negative (low recall) means a cancerous melanoma is missed, which can be fatal. Hence, we must optimize the threshold to maximize Recall.

#### 28. What is the difference between global average pooling and flattening in the classification head?
* **Answer**: Flattening flattens all spatial feature maps ($C \times H \times W$) into a single long vector, resulting in a huge number of linear parameters that depend on input resolution. EfficientNet-B0 uses **Adaptive Average Pooling** before the classifier to compress the spatial dimensions ($H \times W$) into a $1 \times 1$ representation per channel. This makes the head independent of input spatial dimensions and drastically reduces parameter count, protecting against overfitting.

#### 29. Describe the glassmorphism layout and how its CSS is constructed.
* **Answer**: Glassmorphism mimics a frosted glass pane. It is styled in `index.css` via:
  * `background: rgba(20, 24, 32, 0.6)` - Semi-transparent dark background.
  * `backdrop-filter: blur(16px)` - Blurs the radial gradients underneath.
  * `border: 1px solid rgba(255, 255, 255, 0.1)` - Subtle border highlighting the edge.
  * `box-shadow` with deep offsets to create spatial separation.

#### 30. How would you handle continuous learning (retraining) in this project?
* **Answer**: I would set up a feedback pipeline. If user images are logged (with consent), dermatologist-validated cases would be flagged. Once a threshold of new verified samples is reached, an automated pipeline (e.g. using Airflow) would pull the latest weights, run a fine-tuning script, run validation checks against a golden test set to prevent regression, and swap the `model.pth` artifact on the Flask server.

---

## 6. Design Decisions

### Technology Selection Rationale
* **EfficientNet-B0 over ResNet**: Chosen for its high parameter efficiency. It runs extremely fast on a CPU, which was important because we did not want to pay for expensive GPU-backed server hosting.
* **Vite over Create React App (CRA)**: CRA is deprecated and slow. Vite uses esbuild to build the project instantly and offers clean configuration files.
* **Flask over FastAPI**: Although FastAPI is excellent, Flask is simpler and has fewer dependencies, making it perfect for a single-route prediction microservice.

### Alternative Approaches Considered
1. **Frontend-Only Inference (ONNX Web)**:
   * *Alternative*: Compiling to ONNX and doing classification on the client side.
   * *Trade-off*: Good for cost, but exposes the model weights (`model.pth`) directly to anyone inspecting the network tab, which leaks proprietary IP. Keep model on the backend to protect it.
2. **Cloud Functions (Serverless)**:
   * *Alternative*: Deploying Flask prediction to AWS Lambda.
   * *Trade-off*: Zero idle cost, but heavy "cold start" latency due to importing large libraries (torch, torchvision). Standard container hosting is preferred.

### Trade-offs Made
* **Simplification of Output**: Removing raw probability scores and medical terms from the final UI.
  * *Pros*: Avoids confusing patients or causing unnecessary panic.
  * *Cons*: Removes clinical detail that a trained doctor might want to see.

---

## 7. Challenges and Solutions

### Challenge 1: Severe Dataset Imbalance
* *Details*: The HAM10000 dataset contains an overwhelming number of Melanocytic Nevi (common moles) compared to rare conditions like dermatofibroma.
* *Solution*: Applied inverse frequency weights directly in the Cross-Entropy loss function during model training, forcing the model to pay equal attention to minority classes.

### Challenge 2: Cold Starts and GPU-to-CPU Mapping
* *Details*: Model weights saved on CUDA GPUs crash immediately when loaded on CPU-only hosting environments.
* *Solution*: Configured `map_location="cpu"` inside the `torch.load()` command to seamlessly map tensor states to CPU RAM.

### Challenge 3: Responsive Image Previews
* *Details*: File inputs are notoriously difficult to style, and raw uploads can block UI threads.
* *Solution*: Implemented custom hidden input fields triggered by clicking the entire upload area card and used browser-native `URL.createObjectURL` to construct instantaneous preview streams.

---

## 8. Resume-Friendly Explanations

### 30-Second Pitch (HR Screen)
> "I developed a Skin Cancer Detection AI application that classifies skin lesion photographs into cancerous and non-cancerous categories. I fine-tuned a custom-weighted EfficientNet-B0 model in PyTorch on the HAM10000 dataset to address class imbalances. I then deployed the model via a Python Flask microservice connected to a modern, responsive React interface styled with custom CSS glassmorphism."

### 1-Minute Pitch (Technical Screener)
> "I built a dermatological screening application utilizing a React frontend and a PyTorch backend. To overcome severe class imbalances in the HAM10000 dataset, I calculated inverse-frequency class weights for the loss function and trained the model using a two-stage fine-tuning process. I served the model using a stateless Flask API. For the frontend, I built a fast, glassmorphism-themed UI in Vite-React that handles local file streams, displays instant previews, and queries the API asynchronously using Axios, achieving high performance on both CPU hosts and mobile layouts."

### 3-Minute Explanation (Deep Technical Walkthrough)
* **Problem**: Early triage of melanoma is critical, but dataset imbalances lead to models biased towards common moles.
* **Backend ML**: I implemented a PyTorch pipeline fine-tuning an ImageNet-pretrained EfficientNet-B0. I replaced the classifier head to target the 7 HAM10000 lesion categories. I computed class frequencies to weight `nn.CrossEntropyLoss`. I ran stage-1 training (frozen backbone) to stable-start the linear weights, then stage-2 (fully unfrozen) with AdamW and Cosine Annealing.
* **Serving**: I wrote a Flask API returning JSON mappings of the predictions. I applied strict RGB conversion and ImageNet mean/standard deviation normalization to the incoming Pillow image streams. The model runs within a `torch.no_grad()` block to optimize CPU speed.
* **Frontend**: I structured a Vite-React app styled with pure CSS (utilizing variable design tokens and backdrop-filter transitions). It utilizes local ObjectURLs for latency-free previews and updates dynamically with React states.

### Target-Specific Formats
* **HR-Friendly**: Focuses on accessibility, modern design, clean structure, and standard software engineering practices.
* **Technical Interviewer**: Focuses on fine-tuning details (AdamW, Cosine Annealing, Cross-Entropy weights), input normalization, CPU latency considerations, and state synchronization.

---

## 9. Code Walkthrough Guide

### Interview Walkthrough Order
1. **Show the Product first**: Explain the user flow: Selection -> Preview -> Analysis -> Result.
2. **Open the Backend Inference**: Show [app.py](file:///Users/larenpinto/Desktop/oe-mini-project/%20backend/app.py) first to explain how the model is loaded and run on CPU.
3. **Open the Training Pipeline**: Show the Jupyter Notebook [OE_Fixed.ipynb](file:///Users/larenpinto/Desktop/oe-mini-project/OE_Fixed.ipynb) to explain how the weights in `model.pth` were generated.
4. **Open the React Logic**: Walk through [App.jsx](file:///Users/larenpinto/Desktop/oe-mini-project/frontend/src/App.jsx) and describe how Axios posts the multipart payload.

### Core Business Logic Files
* **Model definition & loading**: [app.py:L12-18](file:///Users/larenpinto/Desktop/oe-mini-project/%20backend/app.py#L12-L18)
* **Transform pipeline**: [app.py:L20-24](file:///Users/larenpinto/Desktop/oe-mini-project/%20backend/app.py#L20-L24)
* **Prediction handling**: [app.py:L38-60](file:///Users/larenpinto/Desktop/oe-mini-project/%20backend/app.py#L38-L60)

### Key Snippets to Memorize

#### 1. Flask Prediction Route ([app.py](file:///Users/larenpinto/Desktop/oe-mini-project/%20backend/app.py))
```python
file = request.files["file"]
img = Image.open(file).convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    confidence, pred = torch.max(probabilities, 0)
    pred_idx = int(pred.item())
```

#### 2. Fine-Tuning Stage-2 loop ([OE_Fixed.ipynb](file:///Users/larenpinto/Desktop/oe-mini-project/OE_Fixed.ipynb))
```python
# Unfreeze all layers for subtle fine-tuning
for p in model.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
```

---

## 10. Frequently Forgotten Details

### Hidden Dependencies
* **Pillow (PIL)**: Used implicitly to handle file stream loading.
* **torchvision**: Houses pre-defined transforms and model weights architectures.

### Environment & Run Details
* **Port Mapping**: Flask runs on port **5001** (defined in `app.py` line 63) to avoid conflicts with default systems.
* **Directory structure anomaly**: The backend folder has a leading space: `" backend"`. In terminal commands, you must escape it: `cd \ backend` or `cd " backend"`.

### Key Parameters
* **Image dimensions**: $224 \times 224$ pixels.
* **ImageNet Mean**: `[0.485, 0.456, 0.406]`
* **ImageNet Std**: `[0.229, 0.224, 0.225]`

---

## 11. Red Flags & Defenses

### Red Flag 1: No Authentication/Database (HIPAA/GDPR risk)
* *Interviewer*: "How is patient data kept private? Anyone can upload anything."
* *Defense*: "This is built as a stateless screening utility. We purposefully do not store images or metadata, minimizing our security surface area. No database means zero persistent patient identifier storage, matching 'privacy by design' principles."

### Red Flag 2: Model output is mapped to binary status, losing granularity
* *Interviewer*: "Why map 7 clinical categories to simple 'Cancerous' / 'Non-Cancerous'?"
* *Defense*: "This is a triage app. Explaining dermatological terms (like Actinic Keratoses) to laypersons without a physician present can cause dangerous panic or misinterpretation. We classify high-risk lesions as 'Cancerous' to prompt immediate professional attention."

### Red Flag 3: Synchronous prediction in Flask thread
* *Interviewer*: "Flask's default server blocks threads. If 100 users upload files simultaneously, it will freeze."
* *Defense*: "For development, the Flask debug server is used. In production, this would be served via **Gunicorn** with multiple worker processes, or the inference task would be offloaded asynchronously to a task queue like **Celery** with Redis."

---

## 12. Project Cheat Sheet (Quick Revision)

```
================================================================================
                    SKIN CANCER DETECTION AI CHEAT SHEET
================================================================================

[KEY TECHNOLOGIES]
Frontend: React (v19), Vite, Axios, Custom CSS (Outfit font, Glassmorphism)
Backend : Python, Flask, Flask-CORS, PyTorch (EfficientNet-B0), Pillow

[ARCHITECTURE]
Stateless, client-server structure. Vite feeds inputs to Flask API on port 5001.
Inference done on CPU (using map_location="cpu" load override).

[ML DETAILS (HAM10000)]
Total classes: 7 -> 
  - Malignant: Akiec (0), Bcc (1), Mel (4)    -> mapped to "Cancerous"
  - Benign   : Bkl (2), Df (3), Nv (5), Vasc (6) -> mapped to "Non-Cancerous"
Imbalance Treatment: Inverse-frequency class weights in Cross-Entropy Loss.
Training: 2-stage (Stage 1: freeze backbone; Stage 2: full fine-tune).

[TALKING POINTS]
- "compound scaling in EfficientNet-B0 for high accuracy at low parameters"
- "ImageNet normalisation ensures input matched pretrained feature distribution"
- "torch.no_grad() and unsqueeze(0) for CPU memory and latency optimization"
- "Stateless design respects user privacy and complies with regulations"
================================================================================
```
