## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Installation and Setup](#installation-and-setup)
6. [Usage](#usage)
7. [Contributing](#contributing)

## Project Overview

The **Flight Fare Prediction** project involves the creation of an application that predicts flight prices using a machine learning model trained on **AWS SageMaker**. The application is built using **Streamlit**, allowing users to interactively input flight details and receive fare predictions. This project showcases how to leverage cloud computing for machine learning tasks while providing an intuitive user interface.

## Features

- Predicts flight prices based on various input features such as departure and arrival locations, flight duration, and more.
- User-friendly web application built with Streamlit.
- Model training and deployment conducted using AWS SageMaker.
- Comprehensive performance metrics and evaluation of the predictive model.

## Technologies Used

- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Cloud Platform**: AWS SageMaker
- **Machine Learning Libraries**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn (optional for visualizing data)
- **Development Environment**: Jupyter Notebook or any Python IDE
- **Version Control**: Git

## Dataset

The dataset used in this project consists of flight details and their corresponding prices. It includes various features, such as:

- Departure and Arrival Cities
- Flight Duration
- Airlines
- Date of Journey
- Class of Service
- Number of Stops

You can find the dataset in the `data` folder of this repository or access it from a public dataset source such as [Kaggle](https://www.kaggle.com/datasets).

## Installation and Setup

To set up the Flight Fare Prediction project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/palpratik56/Flightfare-prediction-Sagemaker.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Flightfare-prediction-Sagemaker
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download any necessary datasets**: Ensure that the flight fare dataset is available in the `data` folder as specified in the project files.

## Usage

To run the Streamlit application:

1. Open your terminal and navigate to the project directory.
2. Start the Streamlit application using the following command:
    ```bash
    streamlit run app.py
    ```
3. Open your web browser and navigate to `http://localhost:8501` to access the application.
4. Input the required flight details to receive price predictions.

## Contributing

Contributions to the Flight Fare Prediction project are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/newFeature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/newFeature`).
5. Open a pull request.
