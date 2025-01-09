### README

---

# Lane Detection System for Indian Roads by **Sabbu Sashidhar** ([GitHub](https://github.com/sashidhar498)) and **Aluri Sri Moukthika** ([GitHub](https://github.com/SriMoukthika12))

This repository contains a lane detection system specifically designed for Indian road conditions. It is developed by [**Sabbu Sashidhar**](https://github.com/sashidhar498) and [**Aluri Sri Moukthika**](https://github.com/SriMoukthika12).

The system uses two 80-epoch trained models for morning and night-time detection. Follow the instructions below to set up and run the project.

---

## 📂 **Folder Structure**
The project folder should look like this after downloading the necessary files:

```
project/
│
├── 8batch20epoch.h5                  # Trained model for daytime detection
├── 8batch20epochnight.h5             # Trained model for nighttime detection
├── best.pt                           # YOLOv5 model for object detection
├── keep_left.mp3                     # Audio feedback: Keep left
├── lane_images.py                    # Main script for lane detection
├── lane_images2.py                   # Optimized lane detection with enhancements
├── vehicle_ahead.mp3                 # Audio feedback: Vehicle ahead
├── vehicle_on_left.mp3               # Audio feedback: Vehicle on left
├── video_to_images.py                # Script to convert video to image frames
└── requirements.txt                  # List of required Python libraries
```

---

## 🚀 **Setup Instructions**

### 1. Clone the Repository
Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/your-repository-link.git
cd project
```

### 2. Download Additional Files
Download the required files (`8batch20epoch.h5`, `8batch20epochnight.h5`) from this [Google Drive link](https://drive.google.com/drive/folders/1FPihPe9Cc6lBjZaS4eP6jt-jCzKFSWV6?usp=sharing) and place them in the root directory of the project.

### 3. Install Dependencies
Install the necessary Python libraries by running:
```bash
pip install -r requirements.txt
```

### 4. Convert Video to Image Frames
If you have a video file, convert it into image frames using the `video_to_images.py` script. Replace `videofilename.mp4` with your video file's name:
```bash
python video_to_images.py videofilename.mp4
```

### 5. Run Lane Detection
Run the lane detection script:
```bash
python lane_images.py
```

### **Note**: Ensure your system has an active internet connection when running the scripts to allow required dependencies to download on-demand.

---

## 🛠️ **Requirements**
- Python 3.7+
- TensorFlow and PyTorch frameworks (installed via `requirements.txt`)
- Internet connection for downloading dependencies and files

---

## 💡 **Features**
1. **Daytime and Nighttime Models**:
   - `8batch20epoch.h5` for daytime detection
   - `8batch20epochnight.h5` for nighttime detection

2. **Audio Feedback**:
   - Provides real-time alerts like:
     - "Keep Left"
     - "Vehicle Ahead"
     - "Vehicle on Left"

3. **Video Support**:
   - Converts video to image frames for processing.

4. **Optimized for Indian Roads**:
   - Designed specifically for Indian road conditions, ensuring better performance.

---

## 💻 **Troubleshooting**
1. **Missing Files**:
   - Ensure the trained models and YOLO weights are downloaded from the Google Drive link and placed in the project directory.

2. **Dependency Issues**:
   - Run `pip install -r requirements.txt` again to resolve any missing libraries.

3. **Camera/Frame Issues**:
   - Ensure the input video or frames are correctly processed by verifying the `images/` directory after running `video_to_images.py`.

4. **GPU Usage**:
   - Ensure CUDA drivers are installed for better performance with PyTorch.

---

For issues or contributions, contact us on [GitHub](https://github.com/sashidhar498).  
Happy coding! 😊
