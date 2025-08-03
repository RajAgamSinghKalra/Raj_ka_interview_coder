#!/usr/bin/env python3
"""
Screen Capture & OCR System for Competitive Programming Assistant
Part 3: Screen Capture & OCR

This module provides screen capture functionality and OCR text extraction
to capture competitive programming questions from the screen.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance
import logging
import time
import threading
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import platform
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CaptureConfig:
    """Configuration for screen capture and OCR"""
    # Capture settings
    capture_region: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    capture_full_screen: bool = True
    capture_fps: int = 1  # Frames per second for continuous capture
    
    # OCR settings
    ocr_language: str = 'eng'
    ocr_config: str = '--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
    confidence_threshold: float = 60.0  # Minimum confidence for text detection
    
    # Image preprocessing
    enhance_contrast: bool = True
    enhance_brightness: bool = True
    denoise: bool = True
    binarize: bool = False
    
    # Text processing
    min_text_length: int = 10
    max_text_length: int = 5000
    remove_noise: bool = True
    
    # Question detection
    question_keywords: List[str] = None
    code_keywords: List[str] = None

class ScreenCaptureOCR:
    """Screen capture and OCR system for competitive programming questions"""
    
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.is_capturing = False
        self.capture_thread = None
        self.last_captured_text = ""
        self.text_history = []
        
        # Initialize question detection keywords
        if self.config.question_keywords is None:
            self.config.question_keywords = [
                'problem', 'question', 'task', 'challenge', 'write', 'implement',
                'function', 'algorithm', 'program', 'code', 'solution', 'find',
                'calculate', 'determine', 'compute', 'solve', 'given', 'input',
                'output', 'return', 'print', 'array', 'string', 'integer',
                'maximum', 'minimum', 'sum', 'count', 'reverse', 'sort'
            ]
        
        if self.config.code_keywords is None:
            self.config.code_keywords = [
                'def ', 'function', 'class ', 'if ', 'for ', 'while ', 'return',
                'print', 'input', 'output', 'algorithm', 'complexity', 'time',
                'space', 'O(', 'n)', 'log', 'sort', 'search', 'binary', 'tree',
                'graph', 'dynamic', 'programming', 'recursion', 'iteration'
            ]
        
        # Platform-specific setup
        self.platform = platform.system().lower()
        self._setup_platform()
        
        logger.info("Screen Capture OCR system initialized")
    
    def _setup_platform(self):
        """Setup platform-specific configurations"""
        if self.platform == "windows":
            # Windows-specific setup
            try:
                # Try to find Tesseract installation
                tesseract_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    "tesseract"  # If in PATH
                ]
                
                for path in tesseract_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        logger.info(f"Tesseract found at: {path}")
                        break
                else:
                    logger.warning("Tesseract not found. Please install Tesseract OCR.")
                    
            except Exception as e:
                logger.warning(f"Could not setup Tesseract: {e}")
        
        elif self.platform == "linux":
            # Linux-specific setup
            try:
                # On Linux, Tesseract is usually installed via package manager
                pytesseract.pytesseract.tesseract_cmd = 'tesseract'
                logger.info("Tesseract configured for Linux")
            except Exception as e:
                logger.warning(f"Could not setup Tesseract: {e}")
        
        elif self.platform == "darwin":  # macOS
            # macOS-specific setup
            try:
                # On macOS, Tesseract is usually installed via Homebrew
                pytesseract.pytesseract.tesseract_cmd = 'tesseract'
                logger.info("Tesseract configured for macOS")
            except Exception as e:
                logger.warning(f"Could not setup Tesseract: {e}")
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """Capture screen or specific region"""
        try:
            if region:
                # Capture specific region
                screenshot = ImageGrab.grab(bbox=region)
            elif self.config.capture_region:
                # Use configured region
                screenshot = ImageGrab.grab(bbox=self.config.capture_region)
            else:
                # Capture full screen
                screenshot = ImageGrab.grab()
            
            # Convert PIL Image to OpenCV format
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            return screenshot_cv
            
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhance contrast if enabled
            if self.config.enhance_contrast:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # Enhance brightness if enabled
            if self.config.enhance_brightness:
                # Convert to PIL for brightness enhancement
                pil_image = Image.fromarray(gray)
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(1.2)  # Increase brightness by 20%
                gray = np.array(pil_image)
            
            # Denoise if enabled
            if self.config.denoise:
                gray = cv2.fastNlMeansDenoising(gray)
            
            # Binarize if enabled
            if self.config.binarize:
                _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return gray
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            return image
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from image using OCR"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.config.ocr_language,
                config=self.config.ocr_config
            )
            
            # Clean and process text
            cleaned_text = self._clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return ""
    
    def extract_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text with confidence scores"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Perform OCR with confidence data
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.config.ocr_language,
                config=self.config.ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            total_confidence = 0
            valid_words = 0
            
            for i, conf in enumerate(data['conf']):
                if conf > self.config.confidence_threshold:
                    text_parts.append(data['text'][i])
                    total_confidence += conf
                    valid_words += 1
            
            text = ' '.join(text_parts)
            cleaned_text = self._clean_text(text)
            
            # Calculate average confidence
            avg_confidence = total_confidence / valid_words if valid_words > 0 else 0
            
            return cleaned_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Failed to extract text with confidence: {e}")
            return "", 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean and process extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove noise if enabled
        if self.config.remove_noise:
            # Remove common OCR artifacts
            text = text.replace('|', 'I')  # Common OCR mistake
            text = text.replace('0', 'O')  # In certain contexts
            text = text.replace('1', 'l')  # In certain contexts
        
        # Filter by length
        if len(text) < self.config.min_text_length:
            return ""
        
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        return text.strip()
    
    def detect_question(self, text: str) -> Tuple[bool, float]:
        """Detect if text contains a competitive programming question"""
        if not text:
            return False, 0.0
        
        text_lower = text.lower()
        
        # Count question keywords
        question_score = 0
        for keyword in self.config.question_keywords:
            if keyword.lower() in text_lower:
                question_score += 1
        
        # Count code keywords
        code_score = 0
        for keyword in self.config.code_keywords:
            if keyword.lower() in text_lower:
                code_score += 1
        
        # Calculate confidence score
        total_keywords = len(self.config.question_keywords) + len(self.config.code_keywords)
        confidence = (question_score + code_score) / total_keywords * 100
        
        # Determine if it's a question
        is_question = (
            question_score >= 2 or  # At least 2 question keywords
            (question_score >= 1 and code_score >= 1) or  # Mix of question and code keywords
            confidence >= 30  # High overall confidence
        )
        
        return is_question, confidence
    
    def capture_and_extract(self, region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float, bool]:
        """Capture screen and extract text with question detection"""
        try:
            # Capture screen
            screenshot = self.capture_screen(region)
            if screenshot is None:
                return "", 0.0, False
            
            # Extract text with confidence
            text, confidence = self.extract_text_with_confidence(screenshot)
            
            # Detect if it's a question
            is_question, question_confidence = self.detect_question(text)
            
            # Update history
            if text and text != self.last_captured_text:
                self.last_captured_text = text
                self.text_history.append({
                    'text': text,
                    'confidence': confidence,
                    'is_question': is_question,
                    'question_confidence': question_confidence,
                    'timestamp': time.time()
                })
                
                # Keep only last 10 entries
                if len(self.text_history) > 10:
                    self.text_history.pop(0)
            
            return text, confidence, is_question
            
        except Exception as e:
            logger.error(f"Failed to capture and extract: {e}")
            return "", 0.0, False
    
    def start_continuous_capture(self, callback=None):
        """Start continuous screen capture"""
        if self.is_capturing:
            logger.warning("Continuous capture already running")
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(
            target=self._continuous_capture_loop,
            args=(callback,),
            daemon=True
        )
        self.capture_thread.start()
        logger.info("Started continuous screen capture")
    
    def stop_continuous_capture(self):
        """Stop continuous screen capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        logger.info("Stopped continuous screen capture")
    
    def _continuous_capture_loop(self, callback=None):
        """Continuous capture loop"""
        while self.is_capturing:
            try:
                # Capture and extract
                text, confidence, is_question = self.capture_and_extract()
                
                # Call callback if provided
                if callback and text and is_question:
                    callback(text, confidence, is_question)
                
                # Sleep based on FPS
                time.sleep(1.0 / self.config.capture_fps)
                
            except Exception as e:
                logger.error(f"Error in continuous capture loop: {e}")
                time.sleep(1.0)
    
    def get_text_history(self) -> List[Dict[str, Any]]:
        """Get text capture history"""
        return self.text_history.copy()
    
    def clear_history(self):
        """Clear text capture history"""
        self.text_history.clear()
        self.last_captured_text = ""
    
    def save_screenshot(self, filename: str, region: Optional[Tuple[int, int, int, int]] = None):
        """Save screenshot to file"""
        try:
            screenshot = self.capture_screen(region)
            if screenshot is not None:
                cv2.imwrite(filename, screenshot)
                logger.info(f"Screenshot saved to: {filename}")
            else:
                logger.error("Failed to capture screenshot")
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")

def test_screen_capture_ocr():
    """Test the screen capture and OCR system"""
    logger.info("Testing screen capture and OCR system...")
    
    # Create configuration
    config = CaptureConfig(
        capture_fps=1,
        confidence_threshold=60.0,
        enhance_contrast=True,
        enhance_brightness=True,
        denoise=True
    )
    
    # Create capture system
    capture_system = ScreenCaptureOCR(config)
    
    try:
        # Test single capture
        logger.info("Testing single capture...")
        text, confidence, is_question = capture_system.capture_and_extract()
        
        logger.info(f"Extracted text: {text[:100]}...")
        logger.info(f"Confidence: {confidence:.2f}%")
        logger.info(f"Is question: {is_question}")
        
        # Test question detection
        test_text = "Write a function to find the maximum element in an array. Given an array of integers, return the maximum value."
        is_q, q_conf = capture_system.detect_question(test_text)
        logger.info(f"Test question detection: {is_q} (confidence: {q_conf:.2f}%)")
        
        # Save test screenshot
        capture_system.save_screenshot("test_screenshot.png")
        
        logger.info("Screen capture and OCR test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_screen_capture_ocr() 