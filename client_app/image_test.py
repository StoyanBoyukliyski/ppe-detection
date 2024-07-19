import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import cv2
import time
from PIL import Image

# Load and preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32)
    image = image.transpose(2, 0, 1)  # Convert to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    if image.shape != (1, 3, 640, 640):
        raise ValueError("Image must be 1x3x640x640")
    return image

# Create a Triton HTTP client and send the request
def infer_image_http(image, model_name, url='localhost:8000'):
    triton_client = httpclient.InferenceServerClient(url=url)
    
    inputs = []
    outputs = []
    
    inputs.append(httpclient.InferInput('images', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image)
    
    outputs.append(httpclient.InferRequestedOutput('output0'))
    
    results = triton_client.infer(model_name, inputs, outputs=outputs)
    
    output_data = results.as_numpy('output0')
    return output_data

# Create a Triton gRPC client and send the request
def infer_image_grpc(image, model_name, url='localhost:8001'):
    triton_client = grpcclient.InferenceServerClient(url=url)
    
    inputs = []
    outputs = []
    
    inputs.append(grpcclient.InferInput('images', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image)
    
    outputs.append(grpcclient.InferRequestedOutput('output0'))
    
    results = triton_client.infer(model_name, inputs, outputs=outputs)
    
    output_data = results.as_numpy('output0')
    return output_data

# Process video and send frames to Triton server
def process_video(video_path, output_video_path, model_name, use_grpc=True, url_http='localhost:8000', url_grpc='localhost:8001'):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    out = None
    
    frame_count = 0
    total_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if out is None:
            out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 640))
        
        image = preprocess_image(frame)
        
        start_time = time.time()
        
        if use_grpc:
            output = infer_image_grpc(image, model_name, url_grpc)
        else:
            output = infer_image_http(image, model_name, url_http)
        
        end_time = time.time()
        
        total_time += (end_time - start_time)
        frame_count += 1
        
        #print(f"Inference Output for frame {frame_count}")
        
        # Optionally, you can modify the frame based on the output before writing
        # For simplicity, we're writing the original frame here
        out.write(cv2.resize(frame, (640, 640)))
    
    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    
    if frame_count > 0:
        avg_time_per_frame = total_time / frame_count
        print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
    else:
        print("No frames processed.")

if __name__ == "__main__":
    video_path = 'video.mp4'  # Path to your input video file
    output_video_path = 'output_video.mp4'  # Path to the output video file
    model_name = 'yolov8ppe'  # Model name as specified in the configuration
    
    # Set use_grpc to True to use gRPC, False to use HTTP
    use_grpc = False
    
    process_video(video_path, output_video_path, model_name, use_grpc)

