#!/usr/bin/env python3
"""
Invisible Overlay UI for Competitive Programming Assistant
Part 2: Invisible and Click-through Screen Overlay UI

This module creates a transparent, click-through overlay window that can display
AI responses without being detected by screen recording software or proctoring systems.
"""

import tkinter as tk
from tkinter import ttk
import platform
import sys
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OverlayConfig:
    """Configuration for the invisible overlay"""
    # Window settings
    width: int = 800
    height: int = 600
    x_offset: int = 100
    y_offset: int = 100
    
    # Transparency settings
    alpha: float = 0.95  # 0.0 = fully transparent, 1.0 = fully opaque
    click_through: bool = True
    
    # Display settings
    font_family: str = "Consolas"
    font_size: int = 12
    text_color: str = "#00FF00"  # Green text for visibility
    background_color: str = "#000000"  # Black background
    border_color: str = "#333333"
    
    # Behavior settings
    always_on_top: bool = True
    auto_hide: bool = False
    hide_delay: int = 10  # seconds
    
    # Hotkey settings
    toggle_hotkey: str = "<Control-Alt-O>"  # Ctrl+Alt+O to toggle
    hide_hotkey: str = "<Control-Alt-H>"    # Ctrl+Alt+H to hide
    show_hotkey: str = "<Control-Alt-S>"    # Ctrl+Alt+S to show

class InvisibleOverlay:
    """Invisible overlay window for displaying AI responses"""
    
    def __init__(self, config: OverlayConfig):
        self.config = config
        self.root: Optional[tk.Tk] = None
        self.text_widget: Optional[tk.Text] = None
        self.is_visible = False
        self.is_initialized = False
        
        # Platform-specific settings
        self.platform = platform.system().lower()
        logger.info(f"Initializing overlay for platform: {self.platform}")
        
    def initialize(self):
        """Initialize the overlay window"""
        try:
            # Create root window
            self.root = tk.Tk()
            self.root.title("System Monitor")  # Disguise title
            
            # Configure window properties
            self._configure_window_properties()
            
            # Create UI elements
            self._create_ui_elements()
            
            # Setup hotkeys
            self._setup_hotkeys()
            
            # Platform-specific optimizations
            self._apply_platform_optimizations()
            
            self.is_initialized = True
            logger.info("Overlay initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize overlay: {e}")
            raise
    
    def _configure_window_properties(self):
        """Configure window properties for invisibility"""
        if not self.root:
            return
            
        # Set window size and position
        self.root.geometry(f"{self.config.width}x{self.config.height}+{self.config.x_offset}+{self.config.y_offset}")
        
        # Make window always on top
        if self.config.always_on_top:
            self.root.attributes('-topmost', True)
        
        # Set transparency
        self.root.attributes('-alpha', self.config.alpha)
        
        # Platform-specific window attributes
        if self.platform == "windows":
            self._configure_windows_properties()
        elif self.platform == "linux":
            self._configure_linux_properties()
        elif self.platform == "darwin":  # macOS
            self._configure_macos_properties()
    
    def _configure_windows_properties(self):
        """Configure Windows-specific properties"""
        if not self.root:
            return
            
        try:
            # Import Windows-specific modules
            import ctypes
            from ctypes import wintypes
            
            # Get window handle
            hwnd = self.root.winfo_id()
            
            # Set extended window styles for invisibility
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TRANSPARENT = 0x20
            WS_EX_TOOLWINDOW = 0x80
            
            # Get current extended style
            current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            
            # Add layered and transparent styles
            new_style = current_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
            
            # Set new extended style
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            
            # Set window display affinity to exclude from capture
            WDA_EXCLUDEFROMCAPTURE = 0x11
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            
            logger.info("Applied Windows-specific invisibility properties")
            
        except Exception as e:
            logger.warning(f"Could not apply Windows-specific properties: {e}")
    
    def _configure_linux_properties(self):
        """Configure Linux-specific properties"""
        if not self.root:
            return
            
        try:
            # For Linux, we'll use X11-specific properties if available
            # This is a simplified version - full implementation would require X11 bindings
            logger.info("Linux-specific properties would be applied here")
            
        except Exception as e:
            logger.warning(f"Could not apply Linux-specific properties: {e}")
    
    def _configure_macos_properties(self):
        """Configure macOS-specific properties"""
        if not self.root:
            return
            
        try:
            # For macOS, we'll use Cocoa-specific properties if available
            # This is a simplified version - full implementation would require Cocoa bindings
            logger.info("macOS-specific properties would be applied here")
            
        except Exception as e:
            logger.warning(f"Could not apply macOS-specific properties: {e}")
    
    def _create_ui_elements(self):
        """Create UI elements for the overlay"""
        if not self.root:
            return
            
        # Configure root window appearance
        self.root.configure(bg=self.config.background_color)
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg=self.config.background_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create text widget for displaying responses
        self.text_widget = tk.Text(
            main_frame,
            font=(self.config.font_family, self.config.font_size),
            fg=self.config.text_color,
            bg=self.config.background_color,
            insertbackground=self.config.text_color,
            selectbackground=self.config.border_color,
            relief=tk.FLAT,
            borderwidth=0,
            wrap=tk.WORD,
            state=tk.DISABLED  # Start as read-only
        )
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=self.config.background_color,
            fg=self.config.text_color,
            font=(self.config.font_family, 8)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        logger.info("UI elements created successfully")
    
    def _setup_hotkeys(self):
        """Setup global hotkeys for controlling the overlay"""
        if not self.root:
            return
            
        # Bind hotkeys
        self.root.bind(self.config.toggle_hotkey, lambda e: self.toggle_visibility())
        self.root.bind(self.config.hide_hotkey, lambda e: self.hide())
        self.root.bind(self.config.show_hotkey, lambda e: self.show())
        
        # Bind escape key to hide
        self.root.bind("<Escape>", lambda e: self.hide())
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        
        logger.info("Hotkeys configured successfully")
    
    def _apply_platform_optimizations(self):
        """Apply platform-specific optimizations"""
        if self.platform == "windows":
            # Windows-specific optimizations
            if self.root:
                self.root.attributes('-toolwindow', True)
                self.root.overrideredirect(True)  # Remove window decorations
        elif self.platform == "linux":
            # Linux-specific optimizations
            if self.root:
                self.root.overrideredirect(True)
        elif self.platform == "darwin":
            # macOS-specific optimizations
            if self.root:
                self.root.overrideredirect(True)
    
    def show(self):
        """Show the overlay window"""
        if not self.root or not self.is_initialized:
            return
            
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.is_visible = True
            self.status_var.set("Visible")
            logger.info("Overlay shown")
            
        except Exception as e:
            logger.error(f"Failed to show overlay: {e}")
    
    def hide(self):
        """Hide the overlay window"""
        if not self.root or not self.is_initialized:
            return
            
        try:
            self.root.withdraw()
            self.is_visible = False
            self.status_var.set("Hidden")
            logger.info("Overlay hidden")
            
        except Exception as e:
            logger.error(f"Failed to hide overlay: {e}")
    
    def toggle_visibility(self):
        """Toggle overlay visibility"""
        if self.is_visible:
            self.hide()
        else:
            self.show()
    
    def display_response(self, response: str, clear_previous: bool = True):
        """Display AI response in the overlay"""
        if not self.text_widget:
            return
            
        try:
            # Enable text widget for editing
            self.text_widget.config(state=tk.NORMAL)
            
            # Clear previous content if requested
            if clear_previous:
                self.text_widget.delete(1.0, tk.END)
            
            # Insert new response
            self.text_widget.insert(tk.END, response)
            
            # Scroll to top
            self.text_widget.see(1.0)
            
            # Disable text widget to make it read-only
            self.text_widget.config(state=tk.DISABLED)
            
            # Show overlay if hidden
            if not self.is_visible:
                self.show()
            
            self.status_var.set("Response displayed")
            logger.info("Response displayed in overlay")
            
        except Exception as e:
            logger.error(f"Failed to display response: {e}")
    
    def append_response(self, text: str):
        """Append text to the current response"""
        if not self.text_widget:
            return
            
        try:
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, text)
            self.text_widget.see(tk.END)
            self.text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Failed to append response: {e}")
    
    def clear_display(self):
        """Clear the display"""
        if not self.text_widget:
            return
            
        try:
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.config(state=tk.DISABLED)
            self.status_var.set("Display cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear display: {e}")
    
    def set_position(self, x: int, y: int):
        """Set overlay position"""
        if not self.root:
            return
            
        try:
            self.root.geometry(f"+{x}+{y}")
            logger.info(f"Overlay position set to ({x}, {y})")
            
        except Exception as e:
            logger.error(f"Failed to set position: {e}")
    
    def set_size(self, width: int, height: int):
        """Set overlay size"""
        if not self.root:
            return
            
        try:
            current_geometry = self.root.geometry()
            # Extract current position
            if '+' in current_geometry:
                pos_part = current_geometry.split('+')[1]
                x, y = pos_part.split('+')
                self.root.geometry(f"{width}x{height}+{x}+{y}")
            else:
                self.root.geometry(f"{width}x{height}")
            
            logger.info(f"Overlay size set to {width}x{height}")
            
        except Exception as e:
            logger.error(f"Failed to set size: {e}")
    
    def run(self):
        """Run the overlay main loop"""
        if not self.root or not self.is_initialized:
            logger.error("Overlay not initialized")
            return
            
        try:
            logger.info("Starting overlay main loop")
            self.root.mainloop()
            
        except KeyboardInterrupt:
            logger.info("Overlay stopped by user")
        except Exception as e:
            logger.error(f"Overlay main loop error: {e}")
    
    def destroy(self):
        """Destroy the overlay"""
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
                logger.info("Overlay destroyed")
            except Exception as e:
                logger.error(f"Failed to destroy overlay: {e}")

def test_overlay():
    """Test the invisible overlay"""
    logger.info("Testing invisible overlay...")
    
    # Create configuration
    config = OverlayConfig(
        width=600,
        height=400,
        alpha=0.9,
        click_through=True
    )
    
    # Create overlay
    overlay = InvisibleOverlay(config)
    
    try:
        # Initialize overlay
        overlay.initialize()
        
        # Display test message
        test_response = """AI Assistant Response:

Question: Write a function to find the maximum element in an array.

Solution:
```python
def find_max(arr):
    if not arr:
        return None
    return max(arr)

# Example usage
numbers = [3, 7, 2, 9, 1, 5]
result = find_max(numbers)
print(f"Maximum element: {result}")  # Output: Maximum element: 9
```

Time Complexity: O(n)
Space Complexity: O(1)

Hotkeys:
- Ctrl+Alt+O: Toggle visibility
- Ctrl+Alt+H: Hide overlay
- Ctrl+Alt+S: Show overlay
- Escape: Hide overlay
"""
        
        overlay.display_response(test_response)
        
        # Run overlay
        overlay.run()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        overlay.destroy()

if __name__ == "__main__":
    test_overlay() 