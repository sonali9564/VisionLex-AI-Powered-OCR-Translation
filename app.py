import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st
from googletrans import Translator
import asyncio

# Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Google Translator
translator = Translator()

# Updated dictionary with all supported languages for OCR (Tesseract languages)
ocr_languages = {
    'Afrikaans': 'afr',
    'Albanian': 'sqi',
    'Amharic': 'amh',
    'Arabic': 'ara',
    'Armenian': 'asm',
    'Bengali': 'ben',
    'Bosnian': 'bos',
    'Catalan': 'cat',
    'Chinese (Simplified)': 'chi_sim',
    'Chinese (Traditional)': 'chi_tra',
    'Croatian': 'hrv',
    'Czech': 'ces',
    'Danish': 'dan',
    'Dutch': 'nld',
    'English': 'eng',
    'Esperanto': 'epo',
    'Estonian': 'est',
    'Filipino': 'fil',
    'Finnish': 'fin',
    'French': 'fra',
    'Georgian': 'geo',
    'German': 'deu',
    'Greek': 'ell',
    'Gujarati': 'guj',
    'Haitian Creole': 'hat',
    'Hebrew': 'heb',
    'Hindi': 'hin',
    'Hungarian': 'hun',
    'Icelandic': 'isl',
    'Indonesian': 'ind',
    'Italian': 'ita',
    'Japanese': 'jpn',
    'Javanese': 'jav',
    'Kannada': 'kan',
    'Kazakh': 'kaz',
    'Khmer': 'khm',
    'Korean': 'kor',
    'Kurdish': 'kur',
    'Latvian': 'lav',
    'Lithuanian': 'lit',
    'Malay': 'may',
    'Malayalam': 'mal',
    'Marathi': 'mar',
    'Mongolian': 'mon',
    'Nepali': 'nep',
    'Norwegian': 'nor',
    'Persian': 'fas',
    'Polish': 'pol',
    'Portuguese': 'por',
    'Punjabi': 'pan',
    'Romanian': 'ron',
    'Russian': 'rus',
    'Serbian': 'srp',
    'Sinhalese': 'sin',
    'Slovak': 'slk',
    'Sanskrit': 'san',
    'Slovenian': 'slv',
    'Spanish': 'spa',
    'Swahili': 'swa',
    'Swedish': 'swe',
    'Tamil': 'tam',
    'Telugu': 'tel',
    'Thai': 'tha',
    'Turkish': 'tur',
    'Ukrainian': 'ukr',
    'Vietnamese': 'vie',
    'Welsh': 'wel',
    'Yiddish': 'yid',
    'Zulu': 'zul'
}

# Full list of languages supported by Google Translate API (for translation)
language_list = list(ocr_languages.keys())


# Function to preprocess image for better OCR results
def preprocess_image(image):
    """Preprocess the image to improve text extraction."""
    # Convert image to grayscale
    gray_image = image.convert("L")

    # Apply adaptive thresholding for better contrast
    threshold_image = gray_image.point(lambda p: p > 200 and 255)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(threshold_image)
    enhanced_image = enhancer.enhance(2)

    # Apply median filter for denoising
    filtered_image = enhanced_image.filter(ImageFilter.MedianFilter(3))

    return filtered_image


# Function to perform OCR and extract text from image
def extract_text_from_image(image, lang='eng'):
    """Extract text from the given image using pytesseract with specified language."""
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image, lang=lang, config='--psm 6')
    return text


# Function to translate the extracted text into the selected language (async)
async def translate_text_async(text, target_language):
    """Translate the extracted text into the selected language asynchronously."""
    translated = await translator.translate(text, dest=target_language)
    return translated.text


# Main function to build the Streamlit app
def main():
    st.markdown("""
    <div style="text-align: center;">
        <h1>Welcome to VisionLex 🌍</h1>
        <h5><i>🔍 Transforming Images into Multilingual Insights</i></h5>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("**About this app**"):
        st.write("""
        - VisionLex is an AI-powered app that extracts text from images using OCR and translates it into over 50 languages. 
        - It supports multiple image formats and provides accurate text extraction even in complex images. 
        - The app allows you to select both the OCR language and the translation target language. 
        - With advanced image preprocessing, VisionLex ensures enhanced OCR results. 
        - It’s the ideal tool for turning images into multilingual text insights quickly and efficiently.
        """)

    # Step 1: Language selection for the image (language of the text in the image)
    upload_lang = st.selectbox(
        "Select the language of the text in the image",
        language_list,
        index=0
    )
    # Get the corresponding Tesseract language code
    lang_code = ocr_languages[upload_lang]

    # Step 2: Upload an image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Step 3: Open the uploaded image using PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Step 4: Extract text from the image using the selected language
        extracted_text = extract_text_from_image(image, lang=lang_code)

        # Store extracted text in session state
        if 'extracted_text' not in st.session_state:
            st.session_state.extracted_text = ""

        # Display the extracted text immediately after extraction
        if extracted_text:
            st.subheader("Extracted Text")
            st.write(extracted_text)
            st.session_state.extracted_text = extracted_text
        else:
            st.warning("No text found in the image!")

        # Step 5: Dropdown to select language for translation
        selected_language = st.selectbox("Select language to translate", language_list, index=0)

        # Step 6: Initialize or load previous translations list
        if 'translations' not in st.session_state:
            st.session_state.translations = []

        # Step 7: Add a button to perform translation on click
        if st.button("Convert"):
            # Translate the extracted text into the selected language
            if st.session_state.extracted_text:
                translated_text = asyncio.run(
                    translate_text_async(st.session_state.extracted_text, selected_language.lower()))
                st.session_state.translations.append((selected_language, translated_text))

        # Display all previous translations
        if st.session_state.translations:
            for lang, translation in st.session_state.translations:
                st.subheader(f"Translated Text ({lang})")
                st.write(translation)


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
