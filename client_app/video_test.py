import os
import boto3
import uuid
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import cv2
import time
from PIL import Image
from decimal import Decimal
import requests

# Get EC2 instance type
def get_ec2_instance_type():
    ec2_client = boto3.client('ec2')
    response = ec2_client.describe_instances(InstanceIds = ['i-0f6a99553d6d5374e'])
    instance_type = response['Reservations'][0]['Instances'][0]['InstanceType']
    
    return instance_type

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
def infer_image_http(image, model_name, url):
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
def infer_image_grpc(image, model_name, url):
    triton_client = grpcclient.InferenceServerClient(url=url)
    
    inputs = []
    outputs = []
    
    inputs.append(grpcclient.InferInput('images', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image)
    
    outputs.append(grpcclient.InferRequestedOutput('output0'))
    
    results = triton_client.infer(model_name, inputs, outputs=outputs)
    
    output_data = results.as_numpy('output0')
    return output_data

# Append metrics to DynamoDB
def append_to_dynamodb(ec2_machine_type, average_process_time, frame_rate_per_second, video_length):
    dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_REGION', 'eu-central-1'))
    table = dynamodb.Table('VideoProcessingMetrics')
    
    item = {
        'id': str(uuid.uuid4()),
        'EC2_machine_type': ec2_machine_type,
        'average_process_time': Decimal(str(average_process_time)),  # Convert float to Decimal
        'frame_rate_per_second': Decimal(str(frame_rate_per_second)),  # Convert float to Decimal
        'video_length': Decimal(str(video_length)),  # Convert float to Decimal
        'task_type' : str('object_detection')
    }
    
    table.put_item(Item=item)

# Process video and send frames to Triton server
def process_video(video_path, output_video_path, model_name, use_grpc, url_http, url_grpc, ec2_machine_type):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    out = None

    # Get the total length of the video in seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = frame_count_total / fps
    
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
        
        #print(f"Inference Output for frame {frame_count}: {output}")
        
        # Optionally, you can modify the frame based on the output before writing
        # For simplicity, we're writing the original frame here
        out.write(cv2.resize(frame, (640, 640)))
    
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    
    if frame_count > 0:
        average_process_time = total_time / frame_count
        frame_rate_per_second = frame_count / total_time
        print(f"Average time per frame: {average_process_time:.4f} seconds")
        print(f"Frame rate per second: {frame_rate_per_second:.4f} FPS")
        print(f"Total video length: {video_length:.4f} seconds")
        
        # Append to DynamoDB
        append_to_dynamodb(ec2_machine_type, average_process_time, frame_rate_per_second, video_length)
    else:
        print("No frames processed.")

if __name__ == "__main__":
    video_path = os.getenv('VIDEO_PATH', 'video.mp4')
    output_video_path = os.getenv('OUTPUT_VIDEO_PATH', 'output_video.mp4')
    model_name = os.getenv('MODEL_NAME', 'yolov8ppe')
    use_grpc = os.getenv('USE_GRPC', 'false').lower() == 'true'
    url_http = os.getenv('URL_HTTP', 'localhost:8000')
    url_grpc = os.getenv('URL_GRPC', 'localhost:8001')
    ec2_machine_type = get_ec2_instance_type()
    
    process_video(video_path, output_video_path, model_name, use_grpc, url_http, url_grpc, ec2_machine_type)

