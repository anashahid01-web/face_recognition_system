# 🧑‍💻 Face Recognition System

A Python project that learns faces from a folder of photos and then recognizes them in **images** or in a **live webcam stream**, using `face_recognition` (dlib) and OpenCV.

## ✨ Features
- Encodes known faces from a folder (one photo per person, or one folder per person)
- Recognizes faces in an image and saves an annotated copy
- Live webcam recognition
- Shows the **match distance** for every face (lower means more similar) and labels unmatched faces as `Unknown`
- Adjustable strictness (`tolerance`)
- Environment check that detects the common dlib / NumPy problem
- Command line tool and a demo notebook

## ⚙️ How It Works
1. **Detect** faces in the image with dlib's HOG face detector.
2. **Encode** each face into a 128-number vector with dlib's pretrained face model.
3. **Compare** the vector with the saved vectors of known people using Euclidean distance.
4. If the smallest distance is below the tolerance (default 0.5), the face gets that person's name. Otherwise it is `Unknown`.
5. Draw boxes and labels and save or display the result.

Encodings are stored in `encodings.npz` (plain NumPy arrays, no pickle), so loading the file is safe.

## 🚀 Installation

**1. Clone the repository**
```bash
git clone https://github.com/anashahid01-web/face-recognition-system.git
cd face-recognition-system
```

**2. Install the requirements** (Python 3.9 or newer)
```bash
pip install -r requirements.txt
```
`face_recognition` depends on **dlib**, which can be hard to build on Windows (it needs CMake and C++ build tools). If installation fails, use the prebuilt version instead:
```bash
pip uninstall -y dlib
pip install dlib-bin
pip install face_recognition --no-deps
pip install face_recognition_models click
```

**3. Check the setup**
```bash
python face_recognition_system.py check
```
If you see `Unsupported image type, must be 8bit gray or RGB image` even for normal photos, dlib and NumPy do not match. Installing `dlib-bin` (above), or using `numpy<2` with `opencv-python<4.11`, fixes it.

## ▶️ Usage

**1. Add known faces.** Put photos in `known_faces/` (see `known_faces/README.md`). Only use photos of people who agreed to be included.

**2. Encode them**
```bash
python face_recognition_system.py encode
```

**3. Recognize faces in an image**
```bash
python face_recognition_system.py image path/to/photo.jpg
```
The annotated image is saved to `output/recognized.jpg`.

**4. Live webcam**
```bash
python face_recognition_system.py webcam
```
Press `q` to quit.

You can also follow the steps in `face_recognition_demo.ipynb`.

## 📁 Project Structure
```
face-recognition-system/
├── README.md
├── requirements.txt
├── .gitignore
├── face_recognition_system.py     # encoding, recognition, webcam, command line
├── face_recognition_demo.ipynb    # step-by-step demo
└── known_faces/
    └── README.md                  # how to add photos (photos are not included)
```

## 🔒 Privacy and Responsible Use
Faces are biometric data. This repository does not include any face photos or encodings, and `.gitignore` keeps them out. Only add photos of people who gave their consent, and do not use this project for identity verification or surveillance.

## ⚠️ Limitations
- Accuracy depends on lighting, angle, image quality and the number of photos per person. No accuracy evaluation was performed.
- The HOG detector works best on front-facing faces and can miss small or turned faces, or occasionally detect a face that is not there.
- It compares faces against the photos you provide. It cannot identify anyone who is not in `known_faces/`.

## 🛠️ Tech Stack
Python · face_recognition (dlib) · OpenCV · NumPy · Pillow · Matplotlib · Jupyter Notebook

## 👩‍💻 Author
**Ana Shahid**
[LinkedIn](https://www.linkedin.com/in/ana-shahid-243533180) | [GitHub](https://github.com/anashahid01-web)
