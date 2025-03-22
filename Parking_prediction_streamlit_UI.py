import streamlit as st
import cv2
import os
import asyncio
from ultralytics import YOLO
from PIL import Image

# Define file paths
MODEL_PATH = r"G:\Jupyter Projects\Parking_lane_detection\my_best_yolov8_model.pt"
OUTPUT_DIR = r"G:\Jupyter Projects\Parking_lane_detection"

# Ensure there's an event loop running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Streamlit UI
st.set_page_config(layout="wide")  # Set wide layout
st.title("🚗 Parking Detection using YOLOv8")

# Sidebar for file upload
st.sidebar.header("Upload Media")
option = st.sidebar.radio("Choose Input Type:", ("Image", "Video"))


## Function to process image
def process_image(image_path):
    # Run YOLO detection
    results = model.predict(
        source=image_path,
        save=True,
        conf=0.5,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=1
    )

    # Find the latest detection folder
    detect_runs_dir = os.path.join("runs", "detect")
    if os.path.exists(detect_runs_dir):
        # Get all subdirectories in the runs/detect folder
        subdirs = [os.path.join(detect_runs_dir, d) for d in os.listdir(detect_runs_dir) if os.path.isdir(os.path.join(detect_runs_dir, d))]
        # Sort by creation time (most recent first)
        subdirs.sort(key=os.path.getmtime, reverse=True)
        
        if subdirs:
            latest_detect_dir = subdirs[0]  # Get the most recent detection folder
            # Find the image file in the latest detection folder
            detected_images = [f for f in os.listdir(latest_detect_dir) if f.endswith((".jpg", ".png"))]
            if detected_images:
                output_image_path = os.path.join(latest_detect_dir, detected_images[0])  # Get the detected image path
                print(f"✅ Processed image saved at: {output_image_path}")
                return output_image_path
            else:
                print("❌ No detected image found in the latest folder.")
        else:
            print("❌ No detection folders found in runs/detect.")
    else:
        print("❌ Detection folder (runs/detect) does not exist.")

    return None


## Function to process video (live detection)
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None

    # Create a placeholder for the video frame in the Streamlit UI
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO detection on the frame
        results = model(frame)

        # Count empty and filled spaces
        empty_spaces = 0
        filled_spaces = 0

        # Draw bounding boxes with different colors for different classes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                class_id = int(box.cls)  # Get class ID
                confidence = float(box.conf)  # Get confidence score

                # Assign different colors based on class
                if class_id == 0:  # Assuming class 0 is "car"
                    color = (0, 255, 0)  # Green for cars
                    label = f"Car: {confidence:.2f}"
                    filled_spaces += 1  # Increment filled space count
                elif class_id == 1:  # Assuming class 1 is "empty"
                    color = (0, 0, 255)  # Red for empty spaces
                    label = f"Empty: {confidence:.2f}"
                    empty_spaces += 1  # Increment empty space count
                else:
                    color = (255, 0, 0)  # Blue for other classes (if any)
                    label = f"Other: {confidence:.2f}"

                # Draw bounding box and label with smaller font size
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add a blue box in the upper right corner with total empty and filled spaces
        cv2.rectangle(frame, (frame.shape[1] - 200, 0), (frame.shape[1], 70), (255, 0, 0), -1)  # Blue filled box
        cv2.putText(frame, f"Empty: {empty_spaces}", (frame.shape[1] - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Filled: {filled_spaces}", (frame.shape[1] - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert the frame from BGR to RGB (for Streamlit display)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the placeholder with the current frame
        frame_placeholder.image(frame_rgb, caption="Live Detection", use_container_width=True)

    cap.release()
    st.success("✅ Video processing complete.")


# Image Processing
if option == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_image_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Display original image under the sidebar
        st.sidebar.image(Image.open(temp_image_path), caption="Original Image", use_container_width=True)
        
        # Add a "Start Detection" button in the main UI
        if st.button("Start Detection"):
            # Process the image
            with st.spinner("Detecting objects..."):
                processed_image_path = process_image(temp_image_path)
            
            # Display processed image in the main UI
            if processed_image_path:
                try:
                    processed_image = Image.open(processed_image_path)
                    st.image(processed_image, caption="Detected Image", use_container_width=True)
                    st.success(f"✅ Processed image saved at: {processed_image_path}")
                except Exception as e:
                    st.error(f"Error loading processed image: {e}")

# Video Processing
elif option == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_video_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Display original video under the sidebar
        st.sidebar.video(temp_video_path)
        
        # Add a "Start Detection" button in the main UI
        if st.button("Start Detection"):
            # Process the video
            with st.spinner("Detecting objects in video..."):
                process_video(temp_video_path)
                
# Add developer information and links below the sidebar
st.sidebar.markdown("---")  # Add a horizontal line for separation
st.sidebar.markdown("### Project developed by \n M.Shahjhan Gondal \n  shahjhangondal99@gmail.com")

# LinkedIn and GitHub links
linkedin = "https://linkedin.com/in/muhammad-shahjhan-gondal-493884311"
github = "https://github.com/shahjhan99"

st.sidebar.markdown(f"""
- [LinkedIn]({linkedin})
- [GitHub]({github})
""")