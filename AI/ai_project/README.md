# Intel Image Scene Classification & API Microservice

An end-to-end Machine Learning and Distributed Systems pipeline designed to classify landscape and environmental imagery across six distinct geographical classes using a deep Convolutional Neural Network (CNN). The production framework wraps the final model state inside an isolated, lightweight Flask web service, containerized securely via Docker for platform-independent deployment.

---

## 📊 Core Architecture Stack
* **Base Infrastructure Platform:** Python 3.9-slim (Debian Linux core footprint)
* **Deep Learning Subsystem:** TensorFlow / Keras CNN Framework
* **Inference Delivery Engine:** Flask REST API Web Server 
* **Public Container Registry:** Docker Hub
* **Target Analytical Classes:** Buildings, Forest, Glacier, Mountain, Sea, Street

---

## 📁 Project Directory Layout
All development assets and deployment manifests are structured inside this dedicated folder:

```text
ai project/
├── best_model.h5       # Compiled Deep Learning CNN weights matrix
├── labels.txt          # Sequential classification class mapping
├── train.py            # Code containing dataset preprocessing & training loops
├── inference.py        # Independent client validation utility for local model tests
├── app.py              # Flask server production engine processing POST web streams
├── test.py             # Client script validating real-time web server inferences
├── Dockerfile          # Multi-layered Linux configuration deployment recipe
└── requirements.txt    # Strict version-locked manifest of Python dependencies
```

## ⚙️ Running the Production Environment Locally

You do not need to install Python, TensorFlow, or any extra machine learning libraries on your computer to run this application. Because it is containerized, you can deploy it instantly on any system using Docker.

### Step 1: Spin Up the Container
To pull the compiled, verified container image directly from the public cloud registry and deploy it on your local network interface loops, run the following command in your terminal:

```bash
docker run -d -p 5000:5000 shahzaibkai/intel-image-api:latest
```

### Step 2: Verify the API Endpoint
Once the container status is active and listening on your system boundaries, you can send an image directly to the network wrapper.

Open a secondary terminal, navigate to where you have any test image (e.g., test_image.jpg), and run your automated validation script:

```Bash
python test.py
```

## 🌐 API Endpoint Specifications
The containerized microservice exposes a secure REST interface endpoint for client-side processing:

Endpoint Route: ``` http://localhost:5000/predict ```

HTTP Method:``` POST```

Network Payload Content-Type:``` multipart/form-data```

Required Parameter Key: ```file``` (Value must contain the raw binary string of the image file)

Expected Successful JSON Response Schema:
```JSON
{
  "status": "success",
  "prediction": "mountain",
  "confidence": 0.9684
}
```

## 📝 Developer Academic Credentials
Project Scope: Semester 5 — Artificial Intelligence & Software Engineering Lab Assignment

Student Name: Shahzaib Shah

Source Repository: ```UniversityNotes/AI/ai project/```

Docker Image Fingerprint: ```shahzaibkai/intel-image-api:latest```
