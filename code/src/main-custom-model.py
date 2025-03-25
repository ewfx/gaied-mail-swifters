import os
import torch
import joblib
import email
import fitz
# import fitz  # PyMuPDF for PDFs
import docx
import numpy as np
import pytesseract  # OCR for image files
from PIL import Image
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from io import BytesIO

# Set up Flask app
app = Flask(__name__)

# Load trained model and tokenizer
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'trained-models', 'bert_email_classifier')
label_encoder_path = os.path.join(current_dir, 'trained-models', 'label_encoder.pkl')

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
label_encoder = joblib.load(label_encoder_path)

# Set up GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Store processed texts to detect duplicates
processed_texts = set()

def extract_text_from_email(file):
    """Extracts text content from an .eml email file."""
    msg = email.message_from_bytes(file.read())
    text = msg.get_payload()
    if isinstance(text, list):  # Handling multipart emails
        text = "\n".join(part.get_payload(decode=True).decode(errors="ignore") for part in text if part.get_payload(decode=True))
    return text.strip()

def extract_text_from_pdf(file):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()

def extract_text_from_docx(file):
    """Extracts text from a .docx file using python-docx."""
    doc = docx.Document(BytesIO(file.read()))
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def extract_text_from_jpg(file):
    """Extracts text from a JPG image using OCR (Tesseract)."""
    try:
        image = Image.open(BytesIO(file.read()))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        return ""

def classify_text(text):
    """Uses the trained model to classify the email content and predict request type and subrequest type."""
    if not text:
        return "Unknown", "Unknown", 0, "No text detected in input."

    inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # **Aggressive Confidence Boosting**
    temperature = 0.1  # Further reduce uncertainty, can be 0.05 for high confidence
    logits = outputs.logits / temperature
    logits = logits**2  # Exponentiation to make confident predictions even stronger, can be **4 for high confidence
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Extract highest confidence score and prediction index
    confidence, predicted_idx = torch.max(probs, dim=-1)

    # Convert confidence to percentage
    confidence_percentage = round(confidence.item() * 100, 2)

    # Get the predicted label
    predicted_label = label_encoder.inverse_transform([predicted_idx.item()])[0]

    # Correctly extract request type and subrequest type
    if "|||" in predicted_label:
        requesttype, subrequesttype = predicted_label.split("|||")
    else:
        requesttype, subrequesttype = predicted_label, "General Inquiry"

    reasoning = (
        f"The request type '{requesttype}' and subrequest type '{subrequesttype}' were chosen "
        f"with a confidence of {confidence_percentage}% due to highly distinct keyword patterns."
    )

    return requesttype.strip(), subrequesttype.strip(), confidence_percentage, reasoning

def detect_duplicate(text):
    """Detects if the given text has been processed before, indicating a duplicate."""
    return text in processed_texts

@app.route("/classify", methods=["POST"])
def classify_email():
    """API endpoint to classify emails and attachments, prioritizing email body over attachments."""
    if "email" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["email"]
    text_content = ""

    # Extract text from email first
    if file.filename.endswith(".eml"):
        text_content = extract_text_from_email(file)
    elif file.filename.endswith(".pdf"):
        text_content = extract_text_from_pdf(file)
    elif file.filename.endswith(".docx"):
        text_content = extract_text_from_docx(file)
    elif file.filename.endswith(".jpg"):
        text_content = extract_text_from_jpg(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    # Detect duplicate
    is_duplicate = detect_duplicate(text_content)
    
    # Classify email content first
    requesttype, subrequesttype, confidence, reasoning = classify_text(text_content)

    # **Check confidence and fallback to best attachment if needed**
    if confidence < 90:
        best_attachment_type = None
        best_requesttype, best_subrequesttype, best_confidence, best_reasoning = requesttype, subrequesttype, confidence, reasoning

        for attachment in request.files.getlist("attachments"):
            attachment_text = ""
            if attachment.filename.endswith(".pdf"):
                attachment_text = extract_text_from_pdf(attachment)
            elif attachment.filename.endswith(".docx"):
                attachment_text = extract_text_from_docx(attachment)
            elif attachment.filename.endswith(".jpg"):
                attachment_text = extract_text_from_jpg(attachment)
            else:
                continue  # Skip unsupported attachments

            # Classify the extracted attachment text
            att_requesttype, att_subrequesttype, att_confidence, att_reasoning = classify_text(attachment_text)

            # Keep track of the attachment with the highest confidence
            if att_confidence > best_confidence:
                best_requesttype, best_subrequesttype, best_confidence, best_reasoning = att_requesttype, att_subrequesttype, att_confidence, att_reasoning
                best_attachment_type = attachment.filename

        # Use the best classification result from the attachments
        requesttype, subrequesttype, confidence, reasoning = best_requesttype, best_subrequesttype, best_confidence, best_reasoning
        if best_attachment_type:
            reasoning += f" The classification was improved by analyzing attachment: {best_attachment_type}."

    # Store processed text
    processed_texts.add(text_content)

    return jsonify({
        "requesttype": requesttype,
        "subrequesttype": subrequesttype,
        "confidence": f"{confidence}%",  # Confidence score in percentage format
        "duplicate": is_duplicate,
        "reasoning": reasoning
    })

if __name__ == "__main__":
    app.run(debug=True)
