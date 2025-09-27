# Face Recognition System (SCRFD + ArcFace)

🚀 A real-time **face detection + recognition** project built with:
- [SCRFD](https://arxiv.org/abs/2105.04714) → Fast & accurate **face detector**
- [ArcFace](https://arxiv.org/abs/1801.07698) → State-of-the-art **face embedding extractor**
- **Face Database** → Store and recognize known people
- **OpenCV UI** → Simple live webcam interface
- (Optional) Blink detection for **liveness check** (to reject photos/spoofs)

---

### **NOTE :- Create a known_faces directory in the Project Root directory**

## 📂 Project Structure

Face_Recognition/
│── detection/
│ └── scrfd_detector.py # SCRFD face detector wrapper
|
|── known_faces/
│ 
│── recognition/
│ └── database.py # Stores and matches face embeddings
│
│── liveness/
│ └── blink_detector.py # Basic liveness check (blink detection)
│
│── utils/
│ └── register_person.py # Register a new person into the database
│
│── main.py # Run live face recognition
│── requirements.txt # Dependencies
│── README.md # This file


## ⚙️ Installation

1. Clone the repo:
   git clone https://github.com/HR-Shekhar/Face_Recognition.git
   cd Face_Recognition

2. Create and activate a virtual environment:
    python -m venv ml_env        # suggested python version 3.12.9
    ml_env\Scripts\activate      # On Windows
    source ml_env/bin/activate   # On Linux/Mac
Or use Conda 

3. Install dependencies:
    pip install -r requirements.txt


📝 Usage
1. Register a person

Capture face embeddings and store in the database.

From the project root:
    python -m utils.register_person --name Himanshu --samples 6

2. Run real-time recognition

Start the webcam recognition loop:
    python main.py

---

Detects faces using SCRFD
    ...

Extracts embeddings using ArcFace
    ...

Compares against known database

Draws bounding boxes + names on screen

To quit
    Press `q`
