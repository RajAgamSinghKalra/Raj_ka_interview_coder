#!/usr/bin/env python3
"""
OCR + LLM Integration System for Competitive Programming Assistant
Part 4: OCR + LLM Integration

This module integrates the screen capture/OCR system with the fine-tuned LLM
to automatically generate responses to competitive programming questions.
"""

import logging
import time
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import json
import queue

# Import our modules
from screen_capture_ocr import ScreenCaptureOCR, CaptureConfig
from inference_engine import DirectMLInferenceEngine, InferenceConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for OCR + LLM integration"""
    # OCR settings
    ocr_config: CaptureConfig = None
    
    # LLM settings
    llm_config: InferenceConfig = None
    
    # Integration settings
    auto_process: bool = True
    processing_delay: float = 2.0  # seconds to wait before processing
    max_question_length: int = 2000
    min_question_length: int = 20
    
    # Response settings
    max_response_length: int = 1000
    include_explanation: bool = True
    include_complexity: bool = True
    
    # Performance settings
    batch_processing: bool = False
    max_batch_size: int = 5
    
    # Filtering settings
    confidence_threshold: float = 70.0
    question_confidence_threshold: float = 50.0

class OCRLLMIntegration:
    """Integration system for OCR and LLM processing"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.ocr_system: Optional[ScreenCaptureOCR] = None
        self.llm_engine: Optional[DirectMLInferenceEngine] = None
        self.is_running = False
        self.processing_thread = None
        self.question_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Callbacks
        self.on_question_detected: Optional[Callable] = None
        self.on_response_generated: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'questions_processed': 0,
            'responses_generated': 0,
            'errors': 0,
            'start_time': None
        }
        
        logger.info("OCR + LLM Integration system initialized")
    
    def initialize(self):
        """Initialize the integration system"""
        try:
            # Initialize OCR system
            if self.config.ocr_config is None:
                self.config.ocr_config = CaptureConfig()
            
            self.ocr_system = ScreenCaptureOCR(self.config.ocr_config)
            
            # Initialize LLM engine
            if self.config.llm_config is None:
                self.config.llm_config = InferenceConfig()
            
            self.llm_engine = DirectMLInferenceEngine(self.config.llm_config)
            
            logger.info("OCR + LLM Integration system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize integration system: {e}")
            if self.on_error:
                self.on_error(f"Initialization failed: {e}")
            return False
    
    def start_processing(self):
        """Start the integrated processing system"""
        if self.is_running:
            logger.warning("Processing already running")
            return
        
        if not self.ocr_system or not self.llm_engine:
            logger.error("Systems not initialized")
            return
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start OCR continuous capture
        self.ocr_system.start_continuous_capture(self._on_question_captured)
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("OCR + LLM Integration processing started")
    
    def stop_processing(self):
        """Stop the integrated processing system"""
        self.is_running = False
        
        if self.ocr_system:
            self.ocr_system.stop_continuous_capture()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        logger.info("OCR + LLM Integration processing stopped")
    
    def _on_question_captured(self, text: str, confidence: float, is_question: bool):
        """Callback when OCR detects a question"""
        try:
            # Filter by confidence and question detection
            if (confidence < self.config.confidence_threshold or 
                not is_question or
                len(text) < self.config.min_question_length or
                len(text) > self.config.max_question_length):
                return
            
            # Add to processing queue
            question_data = {
                'text': text,
                'confidence': confidence,
                'timestamp': time.time(),
                'id': self.stats['questions_processed']
            }
            
            self.question_queue.put(question_data)
            
            # Call callback if provided
            if self.on_question_detected:
                self.on_question_detected(question_data)
            
            logger.info(f"Question queued for processing: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error in question capture callback: {e}")
            if self.on_error:
                self.on_error(f"Question capture error: {e}")
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Process questions from queue
                if not self.question_queue.empty():
                    question_data = self.question_queue.get(timeout=1.0)
                    self._process_question(question_data)
                else:
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                if self.on_error:
                    self.on_error(f"Processing loop error: {e}")
                time.sleep(1.0)
    
    def _process_question(self, question_data: Dict[str, Any]):
        """Process a single question"""
        try:
            question_text = question_data['text']
            question_id = question_data['id']
            
            logger.info(f"Processing question {question_id}: {question_text[:50]}...")
            
            # Generate response using LLM
            response = self.llm_engine.generate_response(question_text)
            
            # Extract code from response
            code = self.llm_engine.extract_code(response)
            
            # Format response
            formatted_response = self._format_response(question_text, response, code)
            
            # Create response data
            response_data = {
                'question_id': question_id,
                'question': question_text,
                'response': response,
                'code': code,
                'formatted_response': formatted_response,
                'timestamp': time.time(),
                'processing_time': time.time() - question_data['timestamp']
            }
            
            # Add to response queue
            self.response_queue.put(response_data)
            
            # Update statistics
            self.stats['questions_processed'] += 1
            self.stats['responses_generated'] += 1
            
            # Call callback if provided
            if self.on_response_generated:
                self.on_response_generated(response_data)
            
            logger.info(f"Response generated for question {question_id}")
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            self.stats['errors'] += 1
            if self.on_error:
                self.on_error(f"Question processing error: {e}")
    
    def _format_response(self, question: str, response: str, code: str) -> str:
        """Format the response for display"""
        try:
            formatted_parts = []
            
            # Add question
            formatted_parts.append(f"Question: {question}")
            formatted_parts.append("")
            
            # Add response
            if response and response != code:
                formatted_parts.append("Explanation:")
                formatted_parts.append(response)
                formatted_parts.append("")
            
            # Add code
            if code:
                formatted_parts.append("Solution:")
                formatted_parts.append(f"```python\n{code}\n```")
            
            # Add complexity analysis if enabled
            if self.config.include_complexity:
                complexity = self._analyze_complexity(code)
                if complexity:
                    formatted_parts.append("")
                    formatted_parts.append("Complexity Analysis:")
                    formatted_parts.append(complexity)
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return response
    
    def _analyze_complexity(self, code: str) -> str:
        """Analyze time and space complexity of the code"""
        try:
            if not code:
                return ""
            
            # Simple complexity analysis based on code patterns
            code_lower = code.lower()
            
            time_complexity = "O(n)"  # Default
            space_complexity = "O(1)"  # Default
            
            # Time complexity analysis
            if "for" in code_lower and "range" in code_lower:
                if "for" in code_lower and "for" in code_lower[code_lower.find("for")+3:]:
                    time_complexity = "O(n²)"  # Nested loops
                else:
                    time_complexity = "O(n)"
            
            if "sort" in code_lower:
                time_complexity = "O(n log n)"
            
            if "while" in code_lower:
                time_complexity = "O(n)"
            
            # Space complexity analysis
            if "append" in code_lower or "list" in code_lower:
                space_complexity = "O(n)"
            
            if "recursion" in code_lower or "def" in code_lower and "return" in code_lower:
                space_complexity = "O(n)"  # Recursion stack
            
            return f"Time Complexity: {time_complexity}\nSpace Complexity: {space_complexity}"
            
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            return ""
    
    def get_response(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get the next response from the queue"""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_responses(self) -> List[Dict[str, Any]]:
        """Get all available responses"""
        responses = []
        while not self.response_queue.empty():
            try:
                response = self.response_queue.get_nowait()
                responses.append(response)
            except queue.Empty:
                break
        return responses
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        
        if stats['start_time']:
            stats['uptime'] = time.time() - stats['start_time']
            stats['questions_per_minute'] = (stats['questions_processed'] / stats['uptime']) * 60 if stats['uptime'] > 0 else 0
        
        stats['queue_size'] = self.question_queue.qsize()
        stats['response_queue_size'] = self.response_queue.qsize()
        
        return stats
    
    def clear_queues(self):
        """Clear all queues"""
        while not self.question_queue.empty():
            try:
                self.question_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Queues cleared")
    
    def manual_process_question(self, question: str) -> Optional[Dict[str, Any]]:
        """Manually process a question (bypass OCR)"""
        try:
            if not self.llm_engine:
                logger.error("LLM engine not initialized")
                return None
            
            logger.info(f"Manually processing question: {question[:50]}...")
            
            # Generate response
            response = self.llm_engine.generate_response(question)
            code = self.llm_engine.extract_code(response)
            
            # Format response
            formatted_response = self._format_response(question, response, code)
            
            # Create response data
            response_data = {
                'question_id': -1,  # Manual processing
                'question': question,
                'response': response,
                'code': code,
                'formatted_response': formatted_response,
                'timestamp': time.time(),
                'processing_time': 0.0
            }
            
            # Update statistics
            self.stats['questions_processed'] += 1
            self.stats['responses_generated'] += 1
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in manual processing: {e}")
            self.stats['errors'] += 1
            if self.on_error:
                self.on_error(f"Manual processing error: {e}")
            return None

def test_integration():
    """Test the OCR + LLM integration system"""
    logger.info("Testing OCR + LLM Integration system...")
    
    # Create configuration
    ocr_config = CaptureConfig(
        capture_fps=1,
        confidence_threshold=60.0
    )
    
    llm_config = InferenceConfig(
        model_path="fine_tuned_model",
        max_new_tokens=512
    )
    
    integration_config = IntegrationConfig(
        ocr_config=ocr_config,
        llm_config=llm_config,
        auto_process=True
    )
    
    # Create integration system
    integration = OCRLLMIntegration(integration_config)
    
    try:
        # Initialize system
        if not integration.initialize():
            logger.error("Failed to initialize integration system")
            return
        
        # Test manual processing
        test_question = "Write a function to find the maximum element in an array."
        response_data = integration.manual_process_question(test_question)
        
        if response_data:
            logger.info("Manual processing test successful!")
            logger.info(f"Question: {response_data['question']}")
            logger.info(f"Response: {response_data['formatted_response'][:200]}...")
        else:
            logger.error("Manual processing test failed")
        
        # Get statistics
        stats = integration.get_statistics()
        logger.info(f"Statistics: {stats}")
        
        logger.info("OCR + LLM Integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")

if __name__ == "__main__":
    test_integration() 