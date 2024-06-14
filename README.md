# Candidate Recommendation System API

This repository contains a Candidate Recommendation System with Machine Learning API built using Python and Flask. It is deployed using Google Cloud service that is Cloud Run. This guide will help you get started with setting up, running, and deploying the application.


## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Libraries](#libraries)
3. [Tools](#tools)
4. [Installation](#installation)
5. [Running the Application](#running-the-application)
6. [Docker Setup](#docker-setup)
7. [Deployment](#deployment)

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- [Python](https://nodejs.org/) (3.9)
- [Docker](https://www.docker.com/)
- [Git](https://git-scm.com/)
- [Google Cloud SDK](https://cloud.google.com/sdk)

## Libraries

These are main libraries that are used to create the API service

- [Tensorflow](https://www.tensorflow.org)
- [Natural Languange Toolkit](https://www.nltk.org/)
- [Google Translate](https://pypi.org/project/googletrans/)

## Tools

- **Git:** Version control system.
- **Docker:** Containerization platform.
- **Google Cloud SDK:** CLI tools for interacting with Google Cloud services.

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/CapstoneDicoding/recommendation-system-api.git
    cd recommendation-system-api
    ```

2. **Install dependencies:**

    ```sh
    python3 -m venv venv
    source venv/bin/activate

    pip install -r requirements.txt

    ```

## Running the Application

**Start the application:**

```sh
python3 main.py
```


## Docker Setup

1. **Build the Docker image:**

    ```sh
    docker build -t your-app-name .
    ```

2. **Run the Docker container:**

    ```sh
    docker run -p 8080:8080 your-app-name
    ```

## Deployment

### Cloud Run

1. **Create new Arifacts repository**
    ```sh
    gcloud artifacts repositories create your-repository-name --repository-format=docker --location=asia-southeast2 --async
    ```

2. **Build and push your Docker image to Google Container Registry:**

    ```sh
    gcloud builds submit --tag asia-southeast2-docker.pkg.dev/your-project-id/your-repository-name/your-app-name:tag
    ```

3. **Deploy to Cloud Run:**

    ```sh
    gcloud run deploy --image asia-southeast2-docker.pkg.dev/your-project-id/your-repository-name/your-app-name:tag
    ```

    Follow the prompts to set the region and allow unauthenticated invocations if required.

