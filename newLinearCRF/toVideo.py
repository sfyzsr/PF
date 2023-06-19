import cv2
import os
import CONSTANT
import time

image_folder_path = CONSTANT.SAVE_DIR
number = image_folder_path.split("_")[-1]
output_video_path = 'newLinearCRF/video/' + 'window'+number+"_"+str(time.time())+'.mp4'
fps = 10  # Frames per second


def create_video_from_images(image_folder, output_path, fps):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])
    if not image_files:
        print("No image files found in the specified folder.")
        return
    # image_files[-1]
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

        # Open the source file in binary mode for reading
    with open(os.path.join(image_folder, image_files[-1]), 'rb') as source_file:
        # Read the contents of the source file
        file_data = source_file.read()

    # Open the destination file in binary mode for writing
    with open("newLinearCRF/last/win"+str(number)+" "+str(time.time())+" .png", 'wb') as destination_file:
        # Write the contents to the destination file
        destination_file.write(file_data)



create_video_from_images(image_folder_path, output_video_path, fps)