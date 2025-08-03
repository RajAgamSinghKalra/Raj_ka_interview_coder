#!/usr/bin/env python3
"""
Competitive Programming Assistant - Main Integrated System
Complete invisible AI assistant for competitive programming

This is the main system that integrates all components:
- Part 2: Invisible Overlay UI
- Part 3: Screen Capture & OCR  
- Part 4: OCR + LLM Integration
- Part 5: Answer Display on Overlay
- Part 6: Stealth and Anti-Detection
"""

import logging
import time
import threading
import argparse
import sys
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import our modules
from invisible_overlay import InvisibleOverlay, OverlayConfig
from screen_capture_ocr import ScreenCaptureOCR, CaptureConfig
from ocr_llm_integration import OCRLLMIntegration, IntegrationConfig
from inference_engine import DirectMLInferenceEngine, InferenceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AssistantConfig:
    """Configuration for the complete competitive programming assistant"""
    # Overlay settings
    overlay_width: int = 800
    overlay_height: int = 600
    overlay_x: int = 100
    overlay_y: int = 100
    overlay_alpha: float = 0.95
    
    # OCR settings
    ocr_confidence_threshold: float = 70.0
    ocr_capture_fps: int = 1
    
    # LLM settings
    llm_model_path: str = "fine_tuned_model"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7
    
    # Integration settings
    auto_start: bool = True
    processing_delay: float = 2.0
    
    # Stealth settings
    stealth_mode: bool = True
    low_logging: bool = True
    process_name: str = "SystemMonitor"

class CompetitiveProgrammingAssistant:
    """Complete competitive programming assistant system"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.overlay: Optional[InvisibleOverlay] = None
        self.integration: Optional[OCRLLMIntegration] = None
        self.is_running = False
        self.main_thread = None
        
        # Setup stealth mode
        if self.config.stealth_mode:
            self._setup_stealth()
        
        logger.info("Competitive Programming Assistant initialized")
    
    def _setup_stealth(self):
        """Setup stealth and anti-detection measures"""
        try:
            # Set process name to something innocuous
            if hasattr(os, 'name') and os.name == 'nt':  # Windows
                import ctypes
                ctypes.windll.kernel32.SetConsoleTitleW(self.config.process_name)
            
            # Reduce logging if stealth mode is enabled
            if self.config.low_logging:
                logging.getLogger().setLevel(logging.WARNING)
            
            logger.info("Stealth mode activated")
            
        except Exception as e:
            logger.warning(f"Could not setup stealth mode: {e}")
    
    def initialize(self) -> bool:
        """Initialize all components of the assistant"""
        try:
            logger.info("Initializing Competitive Programming Assistant...")
            
            # Create overlay configuration
            overlay_config = OverlayConfig(
                width=self.config.overlay_width,
                height=self.config.overlay_height,
                x_offset=self.config.overlay_x,
                y_offset=self.config.overlay_y,
                alpha=self.config.overlay_alpha,
                click_through=True,
                always_on_top=True
            )
            
            # Create overlay
            self.overlay = InvisibleOverlay(overlay_config)
            self.overlay.initialize()
            
            # Create OCR configuration
            ocr_config = CaptureConfig(
                capture_fps=self.config.ocr_capture_fps,
                confidence_threshold=self.config.ocr_confidence_threshold,
                enhance_contrast=True,
                enhance_brightness=True,
                denoise=True
            )
            
            # Create LLM configuration
            llm_config = InferenceConfig(
                model_path=self.config.llm_model_path,
                max_new_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                use_directml=True,
                mixed_precision=True
            )
            
            # Create integration configuration
            integration_config = IntegrationConfig(
                ocr_config=ocr_config,
                llm_config=llm_config,
                auto_process=True,
                processing_delay=self.config.processing_delay,
                confidence_threshold=self.config.ocr_confidence_threshold
            )
            
            # Create integration system
            self.integration = OCRLLMIntegration(integration_config)
            
            # Setup callbacks
            self.integration.on_question_detected = self._on_question_detected
            self.integration.on_response_generated = self._on_response_generated
            self.integration.on_error = self._on_error
            
            # Initialize integration
            if not self.integration.initialize():
                logger.error("Failed to initialize integration system")
                return False
            
            logger.info("Competitive Programming Assistant initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {e}")
            return False
    
    def start(self):
        """Start the assistant"""
        if self.is_running:
            logger.warning("Assistant already running")
            return
        
        if not self.overlay or not self.integration:
            logger.error("Assistant not initialized")
            return
        
        try:
            self.is_running = True
            
            # Start integration processing
            self.integration.start_processing()
            
            # Show overlay
            self.overlay.show()
            
            # Display welcome message
            welcome_message = """🤖 Competitive Programming Assistant Ready!

Features:
• Automatic question detection via screen capture
• AI-powered code generation
• Invisible overlay display
• Hotkey controls

Hotkeys:
• Ctrl+Alt+O: Toggle overlay visibility
• Ctrl+Alt+H: Hide overlay
• Ctrl+Alt+S: Show overlay
• Escape: Hide overlay

The assistant is now monitoring your screen for competitive programming questions.
"""
            self.overlay.display_response(welcome_message)
            
            # Start main loop in separate thread
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            logger.info("Competitive Programming Assistant started")
            
        except Exception as e:
            logger.error(f"Failed to start assistant: {e}")
            self.is_running = False
    
    def stop(self):
        """Stop the assistant"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            # Stop integration
            if self.integration:
                self.integration.stop_processing()
            
            # Hide overlay
            if self.overlay:
                self.overlay.hide()
            
            # Wait for main thread
            if self.main_thread:
                self.main_thread.join(timeout=2.0)
            
            logger.info("Competitive Programming Assistant stopped")
            
        except Exception as e:
            logger.error(f"Error stopping assistant: {e}")
    
    def _main_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Check for responses
                response_data = self.integration.get_response(timeout=1.0)
                if response_data:
                    self._display_response(response_data)
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1.0)
    
    def _on_question_detected(self, question_data: Dict[str, Any]):
        """Callback when a question is detected"""
        try:
            question_text = question_data['text']
            confidence = question_data['confidence']
            
            # Display detection message
            detection_message = f"""🔍 Question Detected!

Text: {question_text[:100]}...
Confidence: {confidence:.1f}%

Processing...
"""
            self.overlay.display_response(detection_message, clear_previous=False)
            
            logger.info(f"Question detected: {question_text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in question detection callback: {e}")
    
    def _on_response_generated(self, response_data: Dict[str, Any]):
        """Callback when a response is generated"""
        try:
            question_id = response_data['question_id']
            processing_time = response_data['processing_time']
            
            # Display response
            self._display_response(response_data)
            
            logger.info(f"Response generated for question {question_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in response generation callback: {e}")
    
    def _on_error(self, error_message: str):
        """Callback when an error occurs"""
        try:
            error_display = f"""❌ Error Occurred

{error_message}

The assistant will continue monitoring...
"""
            self.overlay.display_response(error_display, clear_previous=False)
            
            logger.error(f"Assistant error: {error_message}")
            
        except Exception as e:
            logger.error(f"Error in error callback: {e}")
    
    def _display_response(self, response_data: Dict[str, Any]):
        """Display response in overlay"""
        try:
            formatted_response = response_data['formatted_response']
            
            # Add processing info
            processing_time = response_data.get('processing_time', 0)
            info_header = f"⏱️ Generated in {processing_time:.2f}s\n\n"
            
            full_response = info_header + formatted_response
            
            # Display in overlay
            self.overlay.display_response(full_response)
            
        except Exception as e:
            logger.error(f"Error displaying response: {e}")
    
    def manual_process_question(self, question: str) -> Optional[Dict[str, Any]]:
        """Manually process a question"""
        try:
            if not self.integration:
                logger.error("Integration system not initialized")
                return None
            
            # Display processing message
            processing_message = f"""📝 Manual Processing

Question: {question[:100]}...

Generating response...
"""
            self.overlay.display_response(processing_message)
            
            # Process question
            response_data = self.integration.manual_process_question(question)
            
            if response_data:
                # Display response
                self._display_response(response_data)
                return response_data
            else:
                error_message = "❌ Failed to process question. Please try again."
                self.overlay.display_response(error_message)
                return None
                
        except Exception as e:
            logger.error(f"Error in manual processing: {e}")
            error_message = f"❌ Error: {str(e)}"
            self.overlay.display_response(error_message)
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assistant statistics"""
        stats = {
            'is_running': self.is_running,
            'overlay_visible': self.overlay.is_visible if self.overlay else False
        }
        
        if self.integration:
            stats.update(self.integration.get_statistics())
        
        return stats
    
    def run_overlay_only(self):
        """Run only the overlay (for testing)"""
        if not self.overlay:
            logger.error("Overlay not initialized")
            return
        
        try:
            # Show overlay
            self.overlay.show()
            
            # Display test message
            test_message = """🧪 Overlay Test Mode

This is a test of the invisible overlay system.
The assistant is running in overlay-only mode.

Hotkeys:
• Ctrl+Alt+O: Toggle visibility
• Ctrl+Alt+H: Hide overlay
• Ctrl+Alt+S: Show overlay
• Escape: Hide overlay
"""
            self.overlay.display_response(test_message)
            
            # Run overlay
            self.overlay.run()
            
        except Exception as e:
            logger.error(f"Error in overlay-only mode: {e}")
        finally:
            if self.overlay:
                self.overlay.destroy()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Competitive Programming Assistant")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--overlay-only", action="store_true", help="Run overlay only (no OCR/LLM)")
    parser.add_argument("--manual", type=str, help="Manually process a question")
    parser.add_argument("--width", type=int, default=800, help="Overlay width")
    parser.add_argument("--height", type=int, default=600, help="Overlay height")
    parser.add_argument("--x", type=int, default=100, help="Overlay X position")
    parser.add_argument("--y", type=int, default=100, help="Overlay Y position")
    parser.add_argument("--alpha", type=float, default=0.95, help="Overlay transparency")
    parser.add_argument("--model", type=str, default="fine_tuned_model", help="LLM model path")
    parser.add_argument("--no-stealth", action="store_true", help="Disable stealth mode")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AssistantConfig(
        overlay_width=args.width,
        overlay_height=args.height,
        overlay_x=args.x,
        overlay_y=args.y,
        overlay_alpha=args.alpha,
        llm_model_path=args.model,
        stealth_mode=not args.no_stealth
    )
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                import json
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
    
    # Create assistant
    assistant = CompetitiveProgrammingAssistant(config)
    
    try:
        if args.overlay_only:
            # Run overlay only
            if assistant.initialize():
                assistant.run_overlay_only()
            else:
                logger.error("Failed to initialize overlay")
                return 1
        
        elif args.manual:
            # Manual processing mode
            if assistant.initialize():
                assistant.start()
                time.sleep(1)  # Give time to start
                
                response = assistant.manual_process_question(args.manual)
                if response:
                    logger.info("Manual processing completed successfully")
                else:
                    logger.error("Manual processing failed")
                    return 1
                
                # Keep running for a bit to show response
                time.sleep(5)
                assistant.stop()
            else:
                logger.error("Failed to initialize assistant")
                return 1
        
        else:
            # Full assistant mode
            if assistant.initialize():
                assistant.start()
                
                # Keep running until interrupted
                try:
                    while assistant.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                
                assistant.stop()
            else:
                logger.error("Failed to initialize assistant")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Assistant error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 