# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy list of packages needed
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the entire project into the container at /app
COPY ./.env /app/.env
COPY ./app /app/app
COPY ./data /app/data
COPY ./models /app/models

# Make port 5050 available to the world outside this container
EXPOSE 5050

# Run uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5050", "--lifespan", "on", "--loop", "asyncio"]
