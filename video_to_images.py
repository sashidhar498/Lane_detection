import cv2
import os
import sys

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        sys.exit(1)

    frame_count = 0

    while True:
        # Read a frame
        ret, frame = cap.read()

        # Break the loop if no frames are left
        if not ret:
            break

        # Save the frame as an image file
        frame_count += 1
        output_file = os.path.join(output_folder, f"{frame_count}.jpg")
        cv2.imwrite(output_file, frame)
        print(f"Saved: {output_file}")

    # Release the video capture object
    cap.release()
    print(f"Frames saved to folder: {output_folder}")

if __name__ == "__main__":
    # Check if the user provided the correct arguments
    if len(sys.argv) != 2:
        print("Usage: python video_to_images.py input_file_name")
        sys.exit(1)

    # Get input file and output folder from command-line arguments
    input_file = sys.argv[1]

    # Call the function to extract frames
    extract_frames(input_file, "images")
