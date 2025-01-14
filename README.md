# -VisionLex-AI-Powered-OCR-Translation-
**VisionLex: AI-Powered Multilingual OCR** 🌐   VisionLex transforms images into multilingual text insights using advanced OCR and translation technologies. Extract text from 50+ languages, enhance accuracy with preprocessing, and translate into 50+ languages—all via an intuitive Streamlit interface. Fast, reliable, and multilingual!
# VisionLex: Transforming Images into Multilingual Insights 🌍

VisionLex is an AI-powered web application designed to extract text from images and translate it into multiple languages with remarkable accuracy. Built with Python, Streamlit, and Tesseract-OCR, this app combines cutting-edge image processing and translation technologies to deliver efficient and seamless user experiences.

## 🚀 Features
- **Advanced OCR**: Extract text from images in over 50 languages with Tesseract-OCR.
- **Multilingual Translation**: Translate extracted text into more than 50 languages using Google Translate.
- **Image Preprocessing**: Enhance images for improved OCR accuracy, even with complex visuals.
- **User-Friendly Interface**: An intuitive web app built with Streamlit for quick and easy interaction.
- **Support for Multiple Formats**: Works with common image formats such as JPG, JPEG, and PNG.

## 🛠️ Technologies Used
- **Frontend**: [Streamlit](https://streamlit.io/)
- **OCR**: [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) via `pytesseract`
- **Translation**: [Googletrans](https://pypi.org/project/googletrans/)
- **Image Processing**: [Pillow](https://python-pillow.org/)

## 📜 How It Works
1. **Select OCR Language**: Choose the language of the text in the uploaded image.
2. **Upload an Image**: Supports formats like JPG, JPEG, and PNG.
3. **Extract Text**: Leverages Tesseract-OCR with advanced preprocessing for accurate extraction.
4. **Translate Text**: Translate the extracted text into a selected target language.
5. **Multilingual Insights**: View and download the extracted and translated text.

## 📦 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Akanksha4554/visionlex.git
   cd visionlex
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Tesseract-OCR is installed and its path is correctly set:
   - [Download Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🔧 Requirements
- Python 3.12 or later
- Tesseract-OCR installed
- Compatible with Windows (test for Linux/Mac if needed)

## 📁 Dependencies
Refer to [requirements.txt](requirements.txt) for a list of libraries.

## ✨ Contributing
Contributions are welcome! Feel free to submit issues or pull requests for enhancements or bug fixes.

## 📜 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
