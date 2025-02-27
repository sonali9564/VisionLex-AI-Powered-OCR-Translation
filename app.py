import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load environment variables (API keys and other config)
load_dotenv()

# Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Google Generative AI with API key from .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Full list of languages supported by Tesseract OCR (for text extraction)
ocr_languages = {
    'Afrikaans': 'afr',
    'Albanian': 'sqi',
    'Amharic': 'amh',
    'Arabic': 'ara',
    'Armenian': 'arm',
    'Basque': 'eus',
    'Bengali': 'ben',
    'Bosnian': 'bos',
    'Bulgarian': 'bul',
    'Catalan': 'cat',
    'Chinese (Simplified)': 'chi_sim',
    'Chinese (Traditional)': 'chi_tra',
    'Croatian': 'hrv',
    'Czech': 'ces',
    'Danish': 'dan',
    'Dutch': 'nld',
    'English': 'eng',
    'Estonian': 'est',
    'Finnish': 'fin',
    'French': 'fra',
    'Georgian': 'kat',
    'German': 'deu',
    'Greek': 'ell',
    'Gujarati': 'guj',
    'Haitian Creole': 'hat',
    'Hebrew': 'heb',
    'Hindi': 'hin',
    'Hungarian': 'hun',
    'Icelandic': 'isl',
    'Indonesian': 'ind',
    'Irish': 'gle',
    'Italian': 'ita',
    'Japanese': 'jpn',
    'Kannada': 'kan',
    'Kazakh': 'kaz',
    'Korean': 'kor',
    'Latvian': 'lav',
    'Lithuanian': 'lit',
    'Macedonian': 'mac',
    'Malay': 'msa',
    'Malayalam': 'mal',
    'Marathi': 'mar',
    'Mongolian': 'mon',
    'Nepali': 'nep',
    'Norwegian': 'nor',
    'Persian': 'pes',
    'Polish': 'pol',
    'Portuguese': 'por',
    'Punjabi': 'pan',
    'Romanian': 'ron',
    'Russian': 'rus',
    'Serbian': 'srp',
    'Slovak': 'slk',
    'Slovenian': 'slv',
    'Spanish': 'spa',
    'Swahili': 'swa',
    'Swedish': 'swe',
    'Tamil': 'tam',
    'Telugu': 'tel',
    'Turkish': 'tur',
    'Ukrainian': 'ukr',
    'Urdu': 'urd',
    'Uzbek': 'uzb',
    'Vietnamese': 'vie',
    'Welsh': 'wel',
    'Yiddish': 'yid',
}

# Mapping of Tesseract OCR languages to Google Translate API codes
google_translate_langs = {
    'Afrikaans': 'af',
    'Albanian': 'sq',
    'Amharic': 'am',
    'Arabic': 'ar',
    'Armenian': 'hy',
    'Basque': 'eu',
    'Bengali': 'bn',
    'Bosnian': 'bs',
    'Bulgarian': 'bg',
    'Catalan': 'ca',
    'Chinese (Simplified)': 'zh',
    'Chinese (Traditional)': 'zh-TW',
    'Croatian': 'hr',
    'Czech': 'cs',
    'Danish': 'da',
    'Dutch': 'nl',
    'English': 'en',
    'Estonian': 'et',
    'Finnish': 'fi',
    'French': 'fr',
    'Georgian': 'ka',
    'German': 'de',
    'Greek': 'el',
    'Gujarati': 'gu',
    'Haitian Creole': 'ht',
    'Hebrew': 'he',
    'Hindi': 'hi',
    'Hungarian': 'hu',
    'Icelandic': 'is',
    'Indonesian': 'id',
    'Irish': 'ga',
    'Italian': 'it',
    'Japanese': 'ja',
    'Kannada': 'kn',
    'Kazakh': 'kk',
    'Korean': 'ko',
    'Latvian': 'lv',
    'Lithuanian': 'lt',
    'Macedonian': 'mk',
    'Malay': 'ms',
    'Malayalam': 'ml',
    'Marathi': 'mr',
    'Mongolian': 'mn',
    'Nepali': 'ne',
    'Norwegian': 'no',
    'Persian': 'fa',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Punjabi': 'pa',
    'Romanian': 'ro',
    'Russian': 'ru',
    'Serbian': 'sr',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'Spanish': 'es',
    'Swahili': 'sw',
    'Swedish': 'sv',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Urdu': 'ur',
    'Uzbek': 'uz',
    'Vietnamese': 'vi',
    'Welsh': 'cy',
    'Yiddish': 'yi',
}

# Function to preprocess the image to improve OCR results
def preprocess_image(image):
    """Preprocess the image to improve text extraction."""
    gray_image = image.convert("L")  # Convert image to grayscale
    threshold_image = gray_image.point(lambda p: p > 200 and 255)  # Adaptive thresholding
    enhancer = ImageEnhance.Contrast(threshold_image)
    enhanced_image = enhancer.enhance(2)  # Increase contrast
    filtered_image = enhanced_image.filter(ImageFilter.MedianFilter(3))  # Median filter to denoise
    return filtered_image


# Function to clean extracted OCR text
def clean_ocr_text(text):
    """Clean the text extracted by OCR."""
    text = ' '.join(text.split())  # Remove extra spaces
    text = text.replace('“', '"').replace('”', '"')  # Replace curly quotes
    text = text.replace('’', "'").replace('–', '-')  # Replace common symbols
    return text


# Function to perform OCR and extract text from an image
def extract_text_from_image(image, lang='eng'):
    """Extract text from the given image using pytesseract with specified language."""
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image, lang=lang, config='--psm 6')
    return clean_ocr_text(text)


# Function to query Gemini LLM for translation or Q&A using an API request
def get_gemini_response(input_text, prompt):
    """Use Gemini to answer questions or translate the extracted text."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, prompt])
    return response.text


# Function to handle uploaded image
def input_image_setup(uploaded_file, lang):
    """Check if a file has been uploaded, then process it."""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        extracted_text = extract_text_from_image(image, lang=lang)
        return extracted_text, image


# Improved Translation Function with explicit source and target language handling
def improved_translate(input_text, source_lang, target_lang):
    """
    Improved translation logic for better accuracy.
    Uses a clear, structured prompt and ensures smaller, manageable chunks of text.
    """
    # Step 1: Clean the input text to remove unnecessary artifacts
    cleaned_text = clean_ocr_text(input_text)

    # Step 2: Check if the source language and target language are the same
    if source_lang == target_lang:
        return cleaned_text  # If source and target languages are the same, return the original text

    # Step 3: Segment the text into smaller chunks (optional, helps with large texts)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cleaned_text)  # Split by sentence boundaries

    # Step 4: Construct the translation prompt for each chunk
    translated_text = ""

    for sentence in sentences:
        # Clear and structured prompt to provide a better translation quality
        prompt = f"Translate the following text from {source_lang} to {target_lang} in high-quality, formal language:\n\n{sentence}"
        translated_chunk = get_gemini_response(sentence, prompt)  # Send the prompt to Gemini
        translated_text += translated_chunk + " "  # Combine the translated chunks

    return translated_text.strip()


# Main function to build the Streamlit app
def main():
    if "translations" not in st.session_state:
        st.session_state.translations = []
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "selected_lang" not in st.session_state:
        st.session_state.selected_lang = 'English'  # Default language

    st.markdown(""" 
    <div style="text-align: center;">
        <h1>Welcome to VisionLex 🌍</h1>
        <h5><i>🔍 Transforming Images into Multilingual Insights</i></h5>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("**About this app**"):
        st.write(""" 
        - VisionLex uses Tesseract OCR for text extraction and Google Gemini 1.5 Flash for translation and Q&A.
        - The app extracts text from uploaded images and then translates it as well as answers questions based on that text.
        """)

    # Step 1: Language selection for the image (this will not trigger a re-run of Q&A)
    upload_lang = st.selectbox(
        "Select the language of the text in the image",
        list(ocr_languages.keys()),
        index=list(ocr_languages.keys()).index(st.session_state.selected_lang)
    )
    st.session_state.selected_lang = upload_lang  # Save the selected language to session state

    lang_code = ocr_languages[upload_lang]

    # Step 2: Upload an image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Step 3: Extract text from the uploaded image
        extracted_text, image = input_image_setup(uploaded_file, lang_code)

        if extracted_text:
            st.subheader("Extracted Text")
            st.write(extracted_text)

        # Step 4: Translate the extracted text or ask a question
        target_lang = st.selectbox("Select Language for Translation:", list(ocr_languages.keys()))

        # Add language codes to map Tesseract's detected languages to Google Translate's codes
        source_lang_code = google_translate_langs[upload_lang]
        target_lang_code = google_translate_langs[target_lang]

        # Add the Translate Button and only trigger translation when clicked
        if st.button("Translate Text"):
            if source_lang_code == target_lang_code:
                st.warning(
                    f"The source language ({upload_lang}) is the same as the target language ({target_lang}). No translation needed.")
            else:
                # Translate the text using the refined translation function
                translated_text = improved_translate(extracted_text, source_lang_code, target_lang_code)
                st.session_state.translations.append(f"**Translated Text ({target_lang}):**\n\n{translated_text}")

        # Step 5: Display Translated Text History
        if st.session_state.translations:
            st.subheader("Translation History")
            for translation in st.session_state.translations:
                st.markdown(translation)

        # Step 6: Ask a question based on the extracted text (triggered by clicking the "Answer" button)
        question = st.text_input("Ask a question based on the extracted text:")

        # Step 7: Answer Button
        if question:
            answer_button = st.button("Answer")

            # Trigger Q&A when the user clicks the "Answer" button
            if answer_button:
                # Generate the Q&A response
                prompt = (
                    f"Answer the following question in English in a detailed and informative manner, based on the extracted text. "
                    "Only include relevant parts of the extracted text if they are necessary for context or clarification. "
                    "Provide as much detail as possible to ensure the answer is clear and thorough.\n\n"
                    f"Question: {question}\n\nExtracted Text:\n{extracted_text}")

                response = get_gemini_response(extracted_text, prompt)

                # Ensure that Q&A history is numbered sequentially
                question_number = len(st.session_state.qa_history) + 1  # Determine the next question number
                qa_entry = f"**Q{question_number})** {question}\n\n**A{question_number})** {response}"

                # Add the Q&A to the history
                st.session_state.qa_history.append(qa_entry)

        # Step 8: Display Q&A History (Sequential)
        if st.session_state.qa_history:
            st.subheader("Q&A History")
            for qa in st.session_state.qa_history:
                st.markdown(qa)


if __name__ == "__main__":
    main()
