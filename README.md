# Sentiment Analysis Web Application Using NLP

A machine-learning-based web application that analyses English text and classifies it into three sentiment categories: **Positive, Negative, and Neutral**.

The project uses Natural Language Processing (NLP), TF-IDF feature extraction, and Logistic Regression to predict the sentiment of user-provided text.

## Features

- Text input and validation
- Text preprocessing using NLP techniques
- TF-IDF feature extraction
- Sentiment classification using Logistic Regression
- Prediction confidence scores
- Interactive web interface built with Gradio
- Three sentiment categories: Positive, Negative, and Neutral

## Technologies Used

- Python
- Scikit-learn
- NLTK
- Pandas
- NumPy
- TF-IDF Vectorizer
- Logistic Regression
- Gradio
- Joblib
- Google Colab
- Hugging Face Spaces

## Dataset

**Dataset:** Twitter US Airline Sentiment

**Source:** CrowdFlower via Kaggle

[View Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

The dataset contains tweets about six US airlines, labelled as Positive, Negative, or Neutral.

## Model Performance

The model was evaluated using a held-out test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | 76.68% |
| Negative F1-score | 0.86 |
| Neutral F1-score | 0.58 |
| Positive F1-score | 0.67 |
| Test Samples | 2,925 |

## Project Structure

```text
sentiment-analysis-nlp/
│
├── app.py
├── Untitled12.ipynb
├── model.pkl
├── vectorizer.pkl
├── stopwords.pkl
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/ishitarathee06-create/sentiment-analysis-nlp.git
```

2. Navigate to the project directory:

```bash
cd sentiment-analysis-nlp
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

## Application Output

The application predicts sentiment and displays confidence scores for each category.

Example predictions:

| Input Text | Predicted Sentiment |
|---|---|
| I am very happy with the service! | Positive |
| This is the worst experience ever. | Negative |
| The flight leaves at 5 PM. | Neutral |

## Deployment

The application is deployed using **Hugging Face Spaces** with Gradio.

## Author

**Ishita Rathi**

BCA – Bachelor of Computer Applications

Maharaja Surajmal Institute

## Acknowledgement

This project was developed as part of the BCA Summer Training Project (BCA311).
