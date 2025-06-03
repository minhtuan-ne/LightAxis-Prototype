import os
import sys
import time
import tempfile
import cv2
import pygame
import argparse
import numpy as np
from deep_translator import GoogleTranslator
from gtts import gTTS

# OCR Libraries - Using Tesseract OCR
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    OCR_METHOD = "tesseract"
    print("[INFO] Using Tesseract OCR for text recognition")
except ImportError:
    print("[ERROR] Tesseract OCR not found. Please install:")
    print("  pip install pytesseract pillow")
    print("  And install Tesseract system package:")
    print("  - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("  - macOS: brew install tesseract tesseract-lang")
    print("  - Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-vie")
    sys.exit(1)


class CameraTranslator:
    def __init__(self, source_lang='en', target_lang='vi'):
        """Initialize the camera translator with the specified languages"""
        self.source_lang = source_lang  # 'en' for English, 'vi' for Vietnamese
        self.target_lang = target_lang
        self.temp_dir = tempfile.gettempdir()
        self.running = False
        self.camera = None
        
        # Initialize the translator
        self.translator = GoogleTranslator(source=self.get_full_lang_code(source_lang), 
                                          target=self.get_full_lang_code(target_lang))
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Initialize OCR based on available library
        self.setup_ocr()
        
        # Initialize camera
        self.setup_camera()
    
    def setup_ocr(self):
        """Initialize Tesseract OCR"""
        print("[INFO] Setting up Tesseract OCR")
        
        # Test Tesseract installation
        try:
            # Test if Tesseract is accessible
            pytesseract.get_tesseract_version()
            print(f"[INFO] Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            print(f"[ERROR] Tesseract not found: {e}")
            print("Please ensure Tesseract is installed and in your PATH")
            print("Windows users: You may need to set pytesseract.pytesseract.tesseract_cmd")
            sys.exit(1)
        
        # Check available languages
        try:
            available_langs = pytesseract.get_languages(config='')
            print(f"[INFO] Available Tesseract languages: {available_langs}")
            
            # Set up language codes for English and Vietnamese
            self.tesseract_lang = 'eng'  # Default to English
            if 'vie' in available_langs:
                self.tesseract_lang = 'eng+vie'  # Support both if Vietnamese is available
                print("[INFO] Vietnamese language pack detected - supporting both English and Vietnamese")
            else:
                print("[WARNING] Vietnamese language pack not found. Install with:")
                print("  - Ubuntu: sudo apt install tesseract-ocr-vie")
                print("  - macOS: brew install tesseract-lang")
                print("  - Windows: Download from Tesseract releases")
                print("  Falling back to English only")
                
        except Exception as e:
            print(f"[WARNING] Could not check available languages: {e}")
            self.tesseract_lang = 'eng'  # Fallback to English only
    
    def setup_camera(self):
        """Initialize camera"""
        try:
            # Try to open default camera (usually index 0)
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("[ERROR] Could not open camera")
                sys.exit(1)
            
            # Set camera resolution (optional)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            print("[INFO] Camera initialized successfully")
        except Exception as e:
            print(f"[ERROR] Camera setup failed: {e}")
            sys.exit(1)
    
    def get_full_lang_code(self, code):
        """Convert short language code to full code for deep_translator"""
        if code == 'en':
            return 'english'
        elif code == 'vi':
            return 'vietnamese'
        return code
    
    def swap_languages(self):
        """Swap source and target languages"""
        self.source_lang, self.target_lang = self.target_lang, self.source_lang
        # Reinitialize the translator with new language settings
        self.translator = GoogleTranslator(source=self.get_full_lang_code(self.source_lang), 
                                          target=self.get_full_lang_code(self.target_lang))
        source_name = "English" if self.source_lang == "en" else "Vietnamese"
        target_name = "Vietnamese" if self.target_lang == "vi" else "English"
        print(f"\n[SWITCHED] Now translating from {source_name} to {target_name}")
    
    def preprocess_image_for_ocr(self, image_path):
        """Preprocess image to improve OCR accuracy"""
        try:
            # Load image with PIL
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Increase contrast
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)  # Increase sharpness
            
            # Convert to grayscale for better OCR
            image = image.convert('L')
            
            # Apply threshold to get pure black and white
            image = image.point(lambda x: 0 if x < 140 else 255, '1')
            
            # Save preprocessed image for OCR
            preprocessed_path = image_path.replace('.jpg', '_processed.jpg')
            image.save(preprocessed_path)
            
            return preprocessed_path, image
            
        except Exception as e:
            print(f"[WARNING] Image preprocessing failed: {e}")
            return image_path, None
    
    def capture_image(self):
        """Capture image from camera"""
        try:
            ret, frame = self.camera.read()
            if not ret:
                print("[ERROR] Failed to capture image")
                return None, None
            
            # Save the captured image
            image_path = os.path.join(self.temp_dir, f'captured_{time.time()}.jpg')
            cv2.imwrite(image_path, frame)
            
            print(f"[INFO] Image captured and saved to {image_path}")
            return image_path, frame
        except Exception as e:
            print(f"[ERROR] Image capture failed: {e}")
            return None, None
    
    def extract_text_from_image(self, image_path, image_frame=None):
        """Extract text from image using Tesseract OCR"""
        try:
            print("[PROCESSING] Preprocessing image for better OCR...")
            # Preprocess image for better OCR results
            processed_path, processed_image = self.preprocess_image_for_ocr(image_path)
            
            print("[PROCESSING] Extracting text with Tesseract...")
            
            # Use the processed image
            image = Image.open(processed_path)
            
            # Tesseract configuration for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ '
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image, lang=self.tesseract_lang, config=custom_config)
            
            # Clean up the extracted text
            text = text.strip()
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Clean up processed image file
            try:
                os.remove(processed_path)
            except:
                pass
            
            if text:
                print(f"[DEBUG] Raw OCR result: '{text}'")
                return text
            else:
                print("[WARNING] No text extracted from image")
                return None
                
        except Exception as e:
            print(f"[ERROR] Text extraction failed: {e}")
            return None
    
    def translate_text(self, text):
        """Translate text from source language to target language"""
        try:
            return self.translator.translate(text)
        except Exception as e:
            print(f"[ERROR] Translation error: {e}")
            return None
    
    def text_to_speech(self, text, lang):
        """Convert text to speech and play it"""
        try:
            # Generate a unique filename for this speech
            filename = os.path.join(self.temp_dir, f'translated_{time.time()}.mp3')
            
            # Use the proper language code for gTTS
            lang_code = 'en' if lang == 'en' else 'vi'
            
            # Generate speech
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(filename)
            
            # Play the speech using pygame
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.delay(100)
            
            # Clean up the temporary file
            try:
                os.remove(filename)
            except:
                pass
        except Exception as e:
            print(f"[ERROR] Text-to-speech error: {e}")
    
    def show_camera_preview(self):
        """Show live camera preview"""
        print("[INFO] Showing camera preview. Press SPACE to capture, 'q' to quit preview")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Add instructions overlay
            cv2.putText(frame, "Press SPACE to capture image", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit preview", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Preview', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key
                cv2.destroyAllWindows()
                return True  # Capture requested
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False  # Quit requested
    
    def run(self):
        """Run the camera translator in a loop until interrupted"""
        self.running = True
        
        # Print welcome message and instructions
        print("\n===== Camera-Based Image Translator (English ↔ Vietnamese) =====")
        print("Commands:")
        print("  'c' - Show camera preview and capture image")
        print("  's' - Swap languages")
        print("  'q' - Quit")
        
        source_name = "English" if self.source_lang == "en" else "Vietnamese"
        target_name = "Vietnamese" if self.target_lang == "vi" else "English"
        print(f"\nCurrent setting: {source_name} → {target_name}")
        print(f"OCR Method: Tesseract ({self.tesseract_lang})")
        
        try:
            while self.running:
                command = input("\nEnter command (c/s/q): ").lower()
                
                if command == 'q':
                    self.running = False
                    print("Shutting down...")
                    break
                    
                elif command == 's':
                    self.swap_languages()
                    
                elif command == 'c':
                    # Show camera preview and capture
                    capture_requested = self.show_camera_preview()
                    
                    if capture_requested:
                        # Capture image
                        image_path, image_frame = self.capture_image()
                        
                        if image_path:
                            # Extract text from image
                            extracted_text = self.extract_text_from_image(image_path, image_frame)
                            
                            if extracted_text:
                                source_name = "English" if self.source_lang == "en" else "Vietnamese"
                                target_name = "Vietnamese" if self.target_lang == "vi" else "English"
                                
                                print(f"\n[EXTRACTED TEXT ({source_name.upper()})]")
                                print(f"{extracted_text}")
                                
                                # Translate the text
                                translated_text = self.translate_text(extracted_text)
                                
                                if translated_text:
                                    print(f"\n[TRANSLATED TEXT ({target_name.upper()})]")
                                    print(f"{translated_text}")
                                    
                                    # Convert translated text to speech
                                    print(f"\n[SPEAKING] Playing translation in {target_name}...")
                                    self.text_to_speech(translated_text, self.target_lang)
                                else:
                                    print("[ERROR] Translation failed")
                            else:
                                print("[ERROR] No text found in image")
                            
                            # Clean up the image file
                            try:
                                os.remove(image_path)
                            except:
                                pass
                else:
                    print("Invalid command. Use 'c' to capture, 's' to swap languages, or 'q' to quit.")
                    
        except KeyboardInterrupt:
            self.running = False
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description='Camera-based Image Translator (English ↔ Vietnamese)')
    parser.add_argument('--source', '-s', choices=['en', 'vi'], default='en', 
                        help='Source language (en=English, vi=Vietnamese), default: en')
    parser.add_argument('--target', '-t', choices=['en', 'vi'], default='vi',
                        help='Target language (en=English, vi=Vietnamese), default: vi')
    
    args = parser.parse_args()
    
    # Validate that source and target languages are different
    if args.source == args.target:
        print("Error: Source and target languages must be different")
        return
    
    translator = CameraTranslator(source_lang=args.source, target_lang=args.target)
    translator.run()


if __name__ == "__main__":
    main()


# ===== INSTALLATION REQUIREMENTS =====
"""
Required packages:
pip install opencv-python deep-translator gtts pygame pytesseract pillow

Tesseract System Installation:

Windows:
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install the executable
3. Add to PATH or set pytesseract.pytesseract.tesseract_cmd in the code
4. For Vietnamese support, make sure to select language packs during installation

macOS:
brew install tesseract tesseract-lang

Ubuntu/Debian:
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-vie

Testing Tesseract Installation:
tesseract --version
tesseract --list-langs

Usage Examples:
python app.py                    # English to Vietnamese
python app.py -s vi -t en       # Vietnamese to English

Tips for Better OCR:
1. Ensure good lighting when capturing images
2. Hold camera steady and focus on the text
3. Text should be horizontal and clearly visible
4. Avoid shadows and reflections on the text
5. Higher contrast between text and background works better
"""