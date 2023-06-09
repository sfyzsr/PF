import cv2
import os

def create_video_from_images(image_folder, output_path, fps):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    if not image_files:
        print("No image files found in the specified folder.")
        return

    # Read the first image to get its size
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec as needed
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()

    print("Video created successfully.")

# Example usage
image_folder_path = './output/viz_crf_35'
output_video_path = './output/video/video.mp4'
fps = 10  # Frames per second

create_video_from_images(image_folder_path, output_video_path, fps)