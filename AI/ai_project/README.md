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
