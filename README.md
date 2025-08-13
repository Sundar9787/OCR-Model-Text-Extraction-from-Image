# 📄 Smart OCR with Fallback – Streamlit App

This project is an **OCR (Optical Character Recognition) application** built with **Python** and **Streamlit** that intelligently extracts text from images using multiple OCR engines with a fallback mechanism to ensure the best possible result.

---

## 🚀 Features

- Uses **multiple OCR models** for high accuracy:
  - [RapidOCR](https://github.com/RapidAI/RapidOCR)
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - [TrOCR (HuggingFace Transformers)](https://huggingface.co/microsoft/trocr-base-printed)
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- **Confidence-based result selection** – Picks the OCR output with the highest confidence score.
- **Fallback mode** for extracting digits only (useful for numeric data like IDs, prices, etc.).
- Web interface using **Streamlit** for easy usage.
- Preprocessing (grayscale, CLAHE, Gaussian blur, thresholding) for better accuracy.

---

## 🛠 Tech Stack

- **Language**: Python 3.x  
- **Libraries**:
  - `opencv-python`
  - `pytesseract`
  - `rapidocr-onnxruntime`
  - `transformers`
  - `easyocr`
  - `paddleocr`
  - `Pillow`
  - `numpy`
  - `streamlit`

---

## 📂 Project Structure

```
📦 ocr-app
 ┣ 📜 ocr_model.py         # Contains smart_ocr_with_fallback function
 ┣ 📜 app.py               # Streamlit web app
 ┣ 📜 requirements.txt     # Python dependencies
 ┣ 📜 README.md            # Project documentation
 ┗ 📂 sample_images        # (Optional) Sample test images
```

---

## ⚙️ Installation & Setup

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/yourusername/ocr-app.git
cd ocr-app
```

2️⃣ **Create a virtual environment**  
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

4️⃣ **Install Tesseract OCR**  
- **Windows**: Download from [Tesseract OCR GitHub](https://github.com/UB-Mannheim/tesseract/wiki)  
- **Linux**:  
```bash
sudo apt-get install tesseract-ocr
```
- **Mac**:  
```bash
brew install tesseract
```

---

## ▶️ Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Upload an image via the UI and the app will:
1. Run multiple OCR models.
2. Compare confidence scores.
3. Display the best extracted text.

---

## 📷 Demo

![OCR Demo Screenshot](sample_images/demo.png)

---

---

## 🧠 How It Works

1. **Preprocessing** – Images are converted to grayscale, denoised, and thresholded for better OCR accuracy.  
2. **Multi-Engine OCR** – Runs the image through RapidOCR, EasyOCR, PaddleOCR, TrOCR, and Tesseract.  
3. **Confidence Scoring** – Each result has a confidence score, and the highest is chosen.  
4. **Fallback** – If no reliable text is found, tries extracting **digits only**.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
