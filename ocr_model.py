def smart_ocr_with_fallback(image_path, verbose=False):
    import cv2
    import pytesseract
    from rapidocr_onnxruntime import RapidOCR
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import easyocr
    from paddleocr import PaddleOCR
    from PIL import Image
    import numpy as np

    results = []

    # RapidOCR
    try:
        ocr = RapidOCR()
        result, _ = ocr(image_path)
        if result:
            text = ' '.join([r[1] for r in result])
            conf = sum([r[2] for r in result]) / len(result)
            results.append(('RapidOCR', text.strip(), conf))
    except:
        pass

    # EasyOCR
    try:
        reader = easyocr.Reader(['en'], verbose=verbose)
        out = reader.readtext(image_path)
        if out:
            texts = [t for (_, t, c) in out if c > 0.3]
            confs = [c for (_, _, c) in out if c > 0.3]
            if texts:
                results.append(('EasyOCR', ''.join(texts).strip(), np.mean(confs)))
    except:
        pass

    # PaddleOCR
    try:
        paddle = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8,8)).apply(gray)
        out = paddle.ocr(clahe)
        if out and out[0]:
            texts = [line[1][0] for line in out[0]]
            confs = [line[1][1] for line in out[0]]
            if texts:
                results.append(('PaddleOCR', ''.join(texts).strip(), sum(confs)/len(confs)))
    except:
        pass

    # TrOCR
    try:
        image = Image.open(image_path).convert("RGB")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        inputs = processor(image, return_tensors="pt").pixel_values
        ids = model.generate(inputs)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        if text.strip():
            results.append(("TrOCR", text.strip(), 0.85))  # Estimated
    except:
        pass

    # Tesseract (general)
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.'
        text = pytesseract.image_to_string(thresh, config=config).strip()
        data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data['conf'] if c.isdigit()]
        avg_conf = sum(confidences)/len(confidences) if confidences else 0
        if text:
            results.append(('Tesseract-General', text, avg_conf))
    except:
        pass

    # Fallback: Digits only
    try:
        cropped = cv2.imread(image_path)[40:110, 15:180]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = '--psm 7 -c tessedit_char_whitelist=0123456789.'
        fallback_text = pytesseract.image_to_string(binary, config=config).strip()
        if fallback_text and any(c.isdigit() for c in fallback_text):
            results.append(('Tesseract-Digits', fallback_text, 0.90))
    except:
        pass

    # Final decision
    if not results:
        return "❌ No OCR result"

    best = max(results, key=lambda x: x[2])

    if verbose:
        print("\n📋 All OCR Results:")
        for model, txt, conf in results:
            print(f"🔍 {model}: '{txt}' (Confidence: {conf:.3f})")
        print(f"\n✅ FINAL OUTPUT: {best[1]} (via {best[0]})")

    return best[1]
