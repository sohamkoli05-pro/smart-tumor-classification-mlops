# Smart Tumor Classification: From Dataset to Tested API

An educational Machine Learning and MLOps project using Python, scikit-learn, MLflow, and FastAPI.

## Features
- Logistic Regression classification
- GridSearchCV and RandomizedSearchCV
- MLflow experiment tracking
- Accuracy, precision, recall, and F1-score evaluation
- Confusion matrix visualization
- FastAPI prediction endpoint
- API health check

## Technologies Used
- Python 3.12
- scikit-learn
- pandas and NumPy
- MLflow
- FastAPI and Uvicorn
- Matplotlib
- Joblib
- Git and GitHub

## How to Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train the model:

```powershell
python -m src.train
```

Start the API:

```powershell
python -m uvicorn src.api:app --reload
```

Open API documentation in your browser:

`http://127.0.0.1:8000/docs`

## Current Model Results

- Accuracy: 98.25%
- Precision: 98.61%
- Recall: 98.61%
- F1 Score: 98.61%

These results are specific to the current test dataset split.

## Disclaimer

This project is for educational purposes only. It must not be used for actual medical diagnosis or treatment decisions.