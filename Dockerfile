# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy list of packages needed
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip uninstall opencv-python -y
RUN pip install opencv-python-headless

# Copy the entire project into the container at /app
COPY ./.env.example /app/.env
COPY ./app /app/app
COPY ./models /app/models

# Make port 6969 available to the world outside this container
EXPOSE 6969

# Run uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6969", "--lifespan", "on", "--loop", "asyncio"]
