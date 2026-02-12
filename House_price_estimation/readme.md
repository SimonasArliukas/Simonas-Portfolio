# House Price Estimation

A machine learning project to predict house prices using SVR, XGBoost, and LightGBM. 
Includes feature scaling, hyperparameter tuning, and model evaluation.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Contributing](#contributing)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SimonasArliukas/Simonas-Portfolio.git](https://github.com/SimonasArliukas/Simonas-Portfolio.git)
    ```
2.  **Navigate to the project folder:**
    ```bash
    cd House_price_estimation
    ```
3.  **Create and activate a virtual environment:**
    * **Create:**
        ```bash
        python3 -m venv .venv
        ```
    * **Activate (macOS/Linux):**
        ```bash
        source .venv/bin/activate
        ```
    * **Activate (Windows):**
        ```bash
        .venv\Scripts\activate
        ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
## Data

The dataset includes features like:

| Feature        | Description                                             |
|----------------|---------------------------------------------------------|
| MedInc         | Median Income in $10,000 USD                            |
| HouseAge       | Median House Age in years                                |
| AveRooms       | Average number of rooms per household (count)          |
| AveBedrms      | Average number of bedrooms per household (count)       |
| Population     | Total population in the block/district (count)         |
| AveOccup       | Average household occupancy (residents per house)      |
| Latitude       | Block location latitude in decimal degrees             |
| Longitude      | Block location longitude in decimal degrees            |
| MedHouseVal    | Median House Value in $10,000 USD (Target variable)|

## Models

Code includes models like: 

| Model   | Full Name                   | Description                                                      |
|---------|-----------------------------|------------------------------------------------------------------|
| SVR     | Support Vector Regression   | Robust to outliers and effective in high-dimensional spaces.    |
| XGBoost | Extreme Gradient Boosting   | High performance and prevents overfitting through regularization.|
| LightGBM| Light Gradient Boosting     | Optimized for fast training speed and low memory usage.          |

## Contributing

Feel free to fork the project and submit pull requests. Please follow PEP8 style guidelines.


