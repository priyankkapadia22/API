# Use official Python 3.9 image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Uninstall any existing TensorFlow versions to avoid conflicts
RUN pip uninstall -y tensorflow tensorflow-gpu

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Ensure we install the CPU-only version of TensorFlow
RUN pip install --no-cache-dir tensorflow-cpu==2.10.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8000

# Command to run the app
CMD ["python", "app.py"]
