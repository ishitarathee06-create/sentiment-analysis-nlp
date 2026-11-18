
import gradio as gr
import joblib
import re

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

STOP = set(joblib.load("stopwords.pkl"))

# Contractions
CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
    "ain't": "is not",
    "wont": "will not",
    "cant": "can not",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "isnt": "is not",
    "wasnt": "was not",
    "arent": "are not",
    "werent": "were not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "couldnt": "could not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "im": "i am",
    "ive": "i have",
    "youre": "you are"
}

# Emoji replacements
EMOJI = {
    "😊": " happy ",
    "😀": " happy ",
    "😃": " happy ",
    "😄": " happy ",
    "🙂": " happy ",
    "😍": " love ",
    "❤": " love ",
    "👍": " good ",
    "😡": " angry ",
    "😠": " angry ",
    "😞": " sad ",
    "😢": " sad ",
    "😭": " sad ",
    "🙁": " sad ",
    "😔": " sad ",
    "👎": " bad "
}

# Text cleaning function
def clean_text(text):

    t = str(text).lower().replace("’", "'").replace("‘", "'")

    for e, w in EMOJI.items():
        t = t.replace(e, w)

    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = t.replace("#", " ")

    t = re.sub(r"[^a-z'\s]", " ", t)

    t = " ".join(CONTRACTIONS.get(w, w) for w in t.split())

    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'m\b", " am", t)
    t = re.sub(r"'re\b", " are", t)
    t = re.sub(r"'ve\b", " have", t)
    t = re.sub(r"'ll\b", " will", t)
    t = re.sub(r"'d\b", " would", t)
    t = re.sub(r"'s\b", "", t)

    t = re.sub(r"[^a-z\s]", " ", t)

    words = [w for w in t.split() if w not in STOP]

    return " ".join(words)


# Sentiment prediction
def predict_sentiment(text):

    if not text or not text.strip():
        return "Please enter some text.", {}

    if len(text) > 1000:
        return "Text is too long. Maximum 1000 characters.", {}

    cleaned = clean_text(text)

    if not cleaned.strip():
        return "Please enter meaningful text.", {}

    text_vector = vectorizer.transform([cleaned])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    confidence = {
        label.capitalize(): float(prob)
        for label, prob in zip(model.classes_, probabilities)
    }

    return prediction.capitalize(), confidence


# Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("# Sentiment Analyzer")

    gr.Markdown(
        "Classifies text as Positive, Negative or Neutral."
    )

    with gr.Row():

        with gr.Column():

            text_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type something here...",
                lines=5
            )

            submit_btn = gr.Button(
                "Analyze Sentiment",
                variant="primary"
            )

            clear_btn = gr.ClearButton(
                [text_input]
            )

        with gr.Column():

            result = gr.Textbox(
                label="Predicted Sentiment"
            )

            confidence = gr.Label(
                label="Confidence Score",
                num_top_classes=3
            )

    gr.Examples(
        examples=[
            ["I am very happy with the service!"],
            ["This is the worst experience ever."],
            ["The flight leaves at 5 PM."]
        ],
        inputs=text_input
    )

    submit_btn.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[result, confidence]
    )


# Launch application
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
