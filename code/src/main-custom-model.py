import os
import torch
import joblib
import email
import fitz
import docx
import numpy as np
import pytesseract  # OCR for image files
from PIL import Image
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

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

processed_texts = set()

def extract_text_from_email(file):
    """Extracts text content from an .eml email file and properly handles attachments."""
    msg = email.message_from_bytes(file.read())
    email_text = ""
    attachments = []

    # Extract email body text
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if "attachment" not in content_disposition and content_type in ["text/plain", "text/html"]:
                try:
                    email_text += part.get_payload(decode=True).decode(errors="ignore") + "\n"
                except:
                    continue
            elif "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    attachments.append((filename, part.get_payload(decode=True)))
    else:
        email_text = msg.get_payload(decode=True).decode(errors="ignore")

    return email_text.strip(), attachments

def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()

def extract_text_from_docx(file_bytes):
    """Extracts text from a .docx file."""
    doc = docx.Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def extract_text_from_jpg(file_bytes):
    """Extracts text from a JPG image using OCR (Tesseract)."""
    try:
        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except:
        return ""

def classify_text(text):
    """Classifies text using the trained model."""
    if not text:
        return "Unknown", "Unknown", 0, "No text detected."

    inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    temperature = 0.05  
    logits = outputs.logits / temperature
    logits = logits**4  
    probs = torch.nn.functional.softmax(logits, dim=-1)

    confidence, predicted_idx = torch.max(probs, dim=-1)
    confidence_percentage = round(confidence.item() * 100, 2)
    predicted_label = label_encoder.inverse_transform([predicted_idx.item()])[0]

    if "|||" in predicted_label:
        requesttype, subrequesttype = predicted_label.split("|||")
    else:
        requesttype, subrequesttype = predicted_label, "General Inquiry"

    reasoning = (
        f"The request type '{requesttype}' and subrequest type '{subrequesttype}' were chosen "
        f"with a confidence of {confidence_percentage}%."
    )

    return requesttype.strip(), subrequesttype.strip(), confidence_percentage, reasoning

def detect_duplicate(text):
    """Detects if the given text has been processed before."""
    return text in processed_texts

@app.route("/classification", methods=["POST"])
def classify_email():
    """API endpoint to classify emails and attachments."""
    if "email" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["email"]
    text_content = ""
    attachments = []

    if file.filename.endswith(".eml"):
        text_content, attachments = extract_text_from_email(file)
    elif file.filename.endswith(".pdf"):
        text_content = extract_text_from_pdf(file.read())
    elif file.filename.endswith(".docx"):
        text_content = extract_text_from_docx(file.read())
    elif file.filename.endswith(".jpg"):
        text_content = extract_text_from_jpg(file.read())
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    is_duplicate = detect_duplicate(text_content)
    requesttype, subrequesttype, confidence, reasoning = classify_text(text_content)

    # If confidence is low, analyze attachments from .eml or direct attachments
    if confidence < 90 and attachments:
        best_attachment_type = None
        best_requesttype, best_subrequesttype, best_confidence, best_reasoning = requesttype, subrequesttype, confidence, reasoning

        for filename, file_bytes in attachments:
            attachment_text = ""
            if filename.endswith(".pdf"):
                attachment_text = extract_text_from_pdf(file_bytes)
            elif filename.endswith(".docx"):
                attachment_text = extract_text_from_docx(file_bytes)
            elif filename.endswith(".jpg"):
                attachment_text = extract_text_from_jpg(file_bytes)
            else:
                continue  

            att_requesttype, att_subrequesttype, att_confidence, att_reasoning = classify_text(attachment_text)

            if att_confidence > best_confidence:
                best_requesttype, best_subrequesttype, best_confidence, best_reasoning = att_requesttype, att_subrequesttype, att_confidence, att_reasoning
                best_attachment_type = filename

        requesttype, subrequesttype, confidence, reasoning = best_requesttype, best_subrequesttype, best_confidence, best_reasoning
        if best_attachment_type:
            reasoning += f" Classification improved using attachment: {best_attachment_type}."

    processed_texts.add(text_content)

    return jsonify({
        "requesttype": requesttype,
        "subrequesttype": subrequesttype,
        "confidence": f"{confidence}%",
        "duplicate": is_duplicate,
        "reasoning": reasoning
    })

if __name__ == "__main__":
    app.run(debug=True)
