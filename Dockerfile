# Use an official Python runtime as a parent image
FROM python:3.9-slim
# FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at app
ADD . /

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5002
EXPOSE 5000
EXPOSE 8000


# Define environment variable to use the GPU
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


# # Define environment variable
# ENV app=app.py

# Run app.py when the container launches
CMD ["gunicorn", "--workers=2", "--threads=4", "--timeout=90", "app:app"]

