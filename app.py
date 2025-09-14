import streamlit as st
import logging
from transformers import pipeline, BartTokenizer
import pdfplumber
import docx
import torch
from io import BytesIO
import yake
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import base64
import re
import time
import smtplib
from email.mime.text import MIMEText

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for professional, cohesive look
st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #1F2937;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton > button {
        background-color: #1E3A8A;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #3B82F6;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .css-1aumxhk {
        background-color: #E2E8F0;
        padding: 20px;
        border-right: 1px solid #CBD5E1;
    }
    .stTextArea textarea {
        border: 2px solid #CBD5E1;
        border-radius: 6px;
        padding: 10px;
    }
    .stFileUploader > div > div {
        border: 2px dashed #CBD5E1;
        border-radius: 6px;
        padding: 10px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #111827;
        color: white;
        text-align: center;
        padding: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    .reset-button {
        background-color: #DC2626;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .reset-button:hover {
        background-color: #B91C1C;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# List of 20 fun facts (10 human body, 10 computers)
fun_facts = [
    "Fun Facts: The human heart beats around 100,000 times per day.",
    "Fun Facts: Your mouth produces about one litre of saliva each day.",
    "Fun Facts: Your eyes blink around 20 times a minute, over ten million times a year.",
    "Fun Facts: Ears and nose never stop growing.",
    "Fun Facts: Earwax is a type of sweat.",
    "Fun Facts: The body produces 25 million new cells every second.",
    "Fun Facts: Humans can distinguish 1 trillion different odors.",
    "Fun Facts: The gluteus maximus is the largest muscle in the body.",
    "Fun Facts: The brain is more active during sleep than watching TV.",
    "Fun Facts: There are more than 600 muscles in the human body.",
    "Fun Facts: The first computer mouse was made of wood.",
    "Fun Facts: The first computer virus was called 'Creeper' and created in 1971.",
    "Fun Facts: The term 'bug' in computing comes from an actual moth found in a computer.",
    "Fun Facts: The QWERTY keyboard layout was designed to slow down typing to prevent jamming.",
    "Fun Facts: The first hard drive was the size of two refrigerators.",
    "Fun Facts: The first webcam was used to monitor a coffee pot.",
    "Fun Facts: Computers can perform billions of operations per second.",
    "Fun Facts: Over 90% of the world's currency is digital.",
    "Fun Facts: The internet weighs as much as a strawberry (in terms of electron weight).",
    "Fun Facts: A standard US keyboard has 104 keys."
]

# Function to send email with enhanced error handling
def send_email(rating, feedback):
    email_address = "vishalsharma7826353@gmail.com"
    try:
        email_password = st.secrets["EMAIL_PASSWORD"]  # Must match EMAIL_PASSWORD in Streamlit Cloud secrets
    except KeyError as e:
        logger.error(f"Secrets error: {str(e)}")
        st.error(f"Configuration error: Missing EMAIL_PASSWORD in secrets.")
        return

    if rating is None:
        st.error("Please select a star rating before submitting feedback.")
        return
    if not feedback.strip():
        st.error("Please provide feedback text before submitting.")
        return

    msg = MIMEText(f"Rating: {rating} stars\n\nFeedback: {feedback}")
    msg['Subject'] = "App Feedback"
    msg['From'] = email_address
    msg['To'] = email_address

    logger.info(f"Sending email with rating: {rating}, feedback: {feedback}")
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_address, email_password)
        server.sendmail(email_address, email_address, msg.as_string())
        server.quit()
        st.success("Thank you for your feedback!")
        logger.info("Email sent successfully")
    except smtplib.SMTPAuthenticationError as e:
        st.error(f"Email authentication failed: {str(e)}. Please check your app password.")
        logger.error(f"SMTPAuthenticationError: {str(e)}")
    except smtplib.SMTPException as e:
        st.error(f"SMTP error occurred: {str(e)}")
        logger.error(f"SMTPException: {str(e)}")
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}")

# Function to clean text
def clean_text(text):
    """Remove excessive whitespace, non-ASCII, control characters, and invalid tokens."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'[,;]{2,}', ',', text)
    text = text.strip()
    return text if text else " "

# Function to chunk text for summarization
def chunk_text(text, tokenizer, max_tokens=700, max_words=500):
    """Split text into chunks respecting token and word limits."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for word in words:
        current_chunk.append(word)
        current_word_count += 1
        temp_text = ' '.join(current_chunk)
        current_tokens = len(tokenizer(temp_text)['input_ids'])
        
        if current_word_count >= max_words or current_tokens >= max_tokens:
            chunks.append(temp_text)
            current_chunk = []
            current_word_count = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [chunk for chunk in chunks if chunk.strip()]

# Function to extract text from various file types
def extract_text(file, file_type):
    text = ""
    try:
        if file_type == "pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif file_type == "txt":
            text = file.getvalue().decode("utf-8")
        elif file_type == "docx":
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            st.error("Unsupported file type. Please upload a PDF, TXT, or DOCX file.")
            return None
        return clean_text(text)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Load summarizer model with fallback
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.warning(f"Failed to load BART model: {str(e)}. Falling back to T5-small.")
        return pipeline("summarization", model="t5-small", tokenizer="t5-small", device=0 if torch.cuda.is_available() else -1)

# Load tokenizer for chunking
@st.cache_resource
def load_tokenizer():
    return BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Extract keywords using YAKE
def extract_keywords(text, num_keywords=10):
    try:
        kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=num_keywords, features=None)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except Exception as e:
        st.error(f"Error extracting keywords: {str(e)}")
        return []

# Generate image with highlighted keywords in summary
def generate_highlighted_image(summary, keywords):
    img_width, img_height = 800, 600
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    lines = []
    current_line = ""
    words = summary.split()
    for word in words:
        if word.lower() in [k.lower() for k in keywords]:
            if current_line:
                lines.append((current_line, '#1F2937'))
            lines.append((word + " ", '#DC2626'))
            current_line = ""
        else:
            current_line += word + " "
            if len(current_line) > 80:
                lines.append((current_line, '#1F2937'))
                current_line = ""
    if current_line:
        lines.append((current_line, '#1F2937'))
    
    y = 20
    for line, color in lines:
        draw.text((20, y), line, fill=color, font=font)
        y += 20
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return buffered

# Generate audio from summary
def generate_audio(summary):
    try:
        tts = gTTS(summary, lang='en')
        buffered = BytesIO()
        tts.write_to_fp(buffered)
        buffered.seek(0)
        return buffered
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

# Function to reset the page
def reset_page():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main app
def main():
    st.title("Professional Document Summarizer")
    st.write("Upload a document (up to 5 MB) or paste text (up to 2,000 words) to generate summaries, highlighted keywords, and audio output.")
    

    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'img_buffer' not in st.session_state:
        st.session_state.img_buffer = None
    if 'audio_buffer' not in st.session_state:
        st.session_state.audio_buffer = None

    # Sidebar for options
    with st.sidebar:
        st.header("Summary Options")
        summary_length = st.selectbox("Select Summary Length", ["Short", "Medium", "Long"])
        length_map = {"Short": (100, 50), "Medium": (300, 150), "Long": (500, 250)}
        
        st.header("Input Method")
        input_type = st.radio("Choose Input", ["Upload File", "Paste Text"])

    # Load tokenizer for chunking
    tokenizer = load_tokenizer()

    # Main content
    if input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload Document (PDF, TXT, DOCX, max 5 MB)", type=["pdf", "txt", "docx"], key="file_uploader")
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
        if st.session_state.uploaded_file:
            file_size_mb = len(st.session_state.uploaded_file.getvalue()) / (1024 * 1024)  # Convert bytes to MB
            max_file_size_mb = 5
            if file_size_mb > max_file_size_mb:
                st.error(f"File exceeds {max_file_size_mb} MB limit (current: {file_size_mb:.2f} MB). Please upload a smaller file.")
                return
            file_type = st.session_state.uploaded_file.name.split('.')[-1].lower()
            with st.spinner("Extracting text..."):
                text = extract_text(st.session_state.uploaded_file, file_type)
                if text is None:
                    return
                st.session_state.input_text = text
                st.info(f"File size: {file_size_mb:.2f} MB")
        else:
            st.info("Please upload a document to proceed.")
            return

        # Document summarization button
        if st.button("Summarize Document"):
            summarizer = load_summarizer()
            placeholder = st.empty()
            st.session_state.summary = None
            st.session_state.img_buffer = None
            st.session_state.audio_buffer = None
            fact_index = 0

            try:
                text_clean = clean_text(st.session_state.input_text)
                chunks = chunk_text(text_clean, tokenizer)
                if not chunks:
                    st.session_state.summary = "Error: No valid text chunks to summarize."
                else:
                    summaries = []
                    for chunk in chunks:
                        placeholder.info(fun_facts[fact_index % len(fun_facts)])
                        time.sleep(5)
                        fact_index += 1
                        try:
                            summary_chunk = summarizer(chunk, max_length=length_map[summary_length][0], min_length=length_map[summary_length][1], do_sample=False)[0]['summary_text']
                            summaries.append(summary_chunk)
                        except Exception as e:
                            st.warning(f"Failed to summarize chunk: {str(e)}. Skipping.")
                    if not summaries:
                        st.session_state.summary = "Error: Failed to generate any summaries."
                    else:
                        st.session_state.summary = " ".join(summaries)
                        placeholder.info(fun_facts[fact_index % len(fun_facts)])
                        time.sleep(5)
                        fact_index += 1
                        st.session_state.img_buffer = generate_highlighted_image(st.session_state.summary, extract_keywords(st.session_state.summary))
                        placeholder.info(fun_facts[fact_index % len(fun_facts)])
                        time.sleep(5)
                        st.session_state.audio_buffer = generate_audio(st.session_state.summary)
            except Exception as e:
                st.session_state.summary = f"Error generating summary: {str(e)}"

            placeholder.empty()

            if st.session_state.summary and "Error" not in st.session_state.summary:
                st.subheader(f"{summary_length} Summary")
                st.write(st.session_state.summary)
                
                st.subheader("Highlighted Important Words (as Image)")
                st.image(st.session_state.img_buffer.getvalue(), use_column_width=True)
                
                st.subheader("Audio of Summary")
                st.audio(st.session_state.audio_buffer.getvalue(), format="audio/mp3")
                
                st.subheader("Downloads")
                st.download_button("Download Summary Text", st.session_state.summary, file_name="summary.txt")
                st.download_button("Download Highlighted Image", st.session_state.img_buffer.getvalue(), file_name="highlights.png", mime="image/png")
                st.download_button("Download Audio", st.session_state.audio_buffer.getvalue(), file_name="summary.mp3", mime="audio/mp3")

                # Feedback section
                st.subheader("Rate the Summary (Optional)")
                with st.form("feedback_form"):
                    rating = st.feedback("stars")
                    feedback = st.text_area("Feedback/Suggestion")
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        send_email(rating, feedback)

                # Reset button
                if st.button("Reset Page", key="reset_document", help="Clear all inputs and outputs", type="primary"):
                    reset_page()

            else:
                st.error(st.session_state.summary or "Summary generation failed.")

    else:
        text = st.text_area("Paste Text Here (Max 2,000 words)", value=st.session_state.input_text, height=300, key="text_area")
        if text != st.session_state.input_text:
            st.session_state.input_text = text
        if not text.strip():
            st.info("Please paste text to proceed.")
            return
        word_count = len(text.split())
        if word_count > 2000:
            st.error(f"Text exceeds 2,000 words (current: {word_count}). Please use shorter text or upload a document.")
            return
        st.info(f"Text word count: {word_count}")

        # Paragraph summarization button
        if st.button("Generate Summary"):
            summarizer = load_summarizer()
            placeholder = st.empty()
            st.session_state.summary = None
            st.session_state.img_buffer = None
            st.session_state.audio_buffer = None
            fact_index = 0

            try:
                text_clean = clean_text(st.session_state.input_text)
                chunks = chunk_text(text_clean, tokenizer)
                if not chunks:
                    st.session_state.summary = "Error: No valid text chunks to summarize."
                else:
                    summaries = []
                    for chunk in chunks:
                        placeholder.info(fun_facts[fact_index % len(fun_facts)])
                        time.sleep(5)
                        fact_index += 1
                        try:
                            summary_chunk = summarizer(chunk, max_length=length_map[summary_length][0], min_length=length_map[summary_length][1], do_sample=False)[0]['summary_text']
                            summaries.append(summary_chunk)
                        except Exception as e:
                            st.warning(f"Failed to summarize chunk: {str(e)}. Skipping.")
                    if not summaries:
                        st.session_state.summary = "Error: Failed to generate any summaries."
                    else:
                        st.session_state.summary = " ".join(summaries)
                        placeholder.info(fun_facts[fact_index % len(fun_facts)])
                        time.sleep(5)
                        fact_index += 1
                        st.session_state.img_buffer = generate_highlighted_image(st.session_state.summary, extract_keywords(st.session_state.summary))
                        placeholder.info(fun_facts[fact_index % len(fun_facts)])
                        time.sleep(5)
                        st.session_state.audio_buffer = generate_audio(st.session_state.summary)
            except Exception as e:
                st.session_state.summary = f"Error generating summary: {str(e)}"

            placeholder.empty()

            if st.session_state.summary and "Error" not in st.session_state.summary:
                st.subheader(f"{summary_length} Summary")
                st.write(st.session_state.summary)
                
                st.subheader("Highlighted Important Words (as Image)")
                st.image(st.session_state.img_buffer.getvalue(), use_column_width=True)
                
                st.subheader("Audio of Summary")
                st.audio(st.session_state.audio_buffer.getvalue(), format="audio/mp3")
                
                st.subheader("Downloads")
                st.download_button("Download Summary Text", st.session_state.summary, file_name="summary.txt")
                st.download_button("Download Highlighted Image", st.session_state.img_buffer.getvalue(), file_name="highlights.png", mime="image/png")
                st.download_button("Download Audio", st.session_state.audio_buffer.getvalue(), file_name="summary.mp3", mime="audio/mp3")

                # Feedback section
                st.subheader("Rate the Summary (Optional)")
                with st.form("feedback_form"):
                    rating = st.feedback("stars")
                    feedback = st.text_area("Feedback/Suggestion")
                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        send_email(rating, feedback)

                # Reset button
                if st.button("Reset Page", key="reset_paragraph", help="Clear all inputs and outputs", type="primary"):
                    reset_page()

            else:
                st.error(st.session_state.summary or "Summary generation failed.")

    # Highlighted notes with warning colors
    st.error("Input Limits: Pasted text ≤ 2,000 words; documents ≤ 5 MB.", icon="🚨")
    st.warning("File Support: Only PDF, TXT, and DOCX formats are supported.", icon="⚠️")
    st.info("Summary Lengths: Short (~100 words), Medium (~300), Long (~500). Large documents are chunked for processing.", icon="ℹ️")

    # Footer
    st.markdown('<div class="footer">© 2025 Professional Summarizer App | Built by Vishal Sharma</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()