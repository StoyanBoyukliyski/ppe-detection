# Use the NVIDIA Triton Inference Server base image
FROM nvcr.io/nvidia/tritonserver:23.04-py3

# Create a directory for the model repository
RUN mkdir -p /models/yolov8ppe/1

# Copy the model files into the container
COPY model_repository/yolov8ppe/config.pbtxt /models/yolov8ppe/
COPY model_repository/yolov8ppe/1/model.onnx /models/yolov8ppe/1/

# Expose the Triton server ports
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Set the entrypoint to run Triton Inference Server
ENTRYPOINT ["tritonserver", "--model-repository=/models"]
