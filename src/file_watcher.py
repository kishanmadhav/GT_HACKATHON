"""
File Watcher Module
Monitors input folder for new files and triggers the processing pipeline
"""

import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from typing import Callable, List
from datetime import datetime


class DataFileHandler(FileSystemEventHandler):
    """
    Handles file system events for new data files
    
    Triggers the processing pipeline when new CSV/JSON/Parquet files
    are detected in the watched directory
    """
    
    SUPPORTED_EXTENSIONS = ['.csv', '.json', '.parquet']
    
    def __init__(self, callback: Callable[[str], None], 
                 cooldown_seconds: int = 5):
        """
        Initialize the file handler
        
        Args:
            callback: Function to call when a new file is detected
            cooldown_seconds: Minimum seconds between processing same file
        """
        self.callback = callback
        self.cooldown = cooldown_seconds
        self.processed_files = {}  # Track recently processed files
        self.pending_files = {}    # Track files being written
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        if self._is_supported_file(file_path):
            # Wait for file to finish writing
            self._schedule_processing(file_path)
    
    def on_modified(self, event):
        """Handle file modification events (for files still being written)"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        if self._is_supported_file(file_path):
            # Update pending status
            self.pending_files[file_path] = datetime.now()
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file extension is supported"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def _schedule_processing(self, file_path: str):
        """Schedule file processing after ensuring it's completely written"""
        self.pending_files[file_path] = datetime.now()
        
        # Wait and check if file is stable
        time.sleep(2)  # Initial wait
        
        # Check file size stability
        try:
            initial_size = os.path.getsize(file_path)
            time.sleep(1)
            final_size = os.path.getsize(file_path)
            
            if initial_size != final_size:
                # File still being written, wait more
                time.sleep(3)
        except OSError:
            return  # File might have been deleted
        
        # Check cooldown
        if file_path in self.processed_files:
            last_processed = self.processed_files[file_path]
            elapsed = (datetime.now() - last_processed).total_seconds()
            if elapsed < self.cooldown:
                print(f"⏳ Skipping {file_path} (cooldown: {self.cooldown - elapsed:.1f}s remaining)")
                return
        
        # Process the file
        print(f"\n📁 New file detected: {file_path}")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            self.callback(file_path)
            self.processed_files[file_path] = datetime.now()
        except Exception as e:
            print(f"❌ Error processing file: {e}")
        
        # Clean up
        if file_path in self.pending_files:
            del self.pending_files[file_path]


class FolderWatcher:
    """
    Watches a folder for new data files
    
    Uses the watchdog library for cross-platform file system monitoring
    """
    
    def __init__(self, watch_path: str, callback: Callable[[str], None]):
        """
        Initialize the folder watcher
        
        Args:
            watch_path: Path to the folder to watch
            callback: Function to call when new files are detected
        """
        self.watch_path = os.path.abspath(watch_path)
        self.callback = callback
        self.observer = None
        self.handler = None
        
        # Create watch directory if it doesn't exist
        os.makedirs(self.watch_path, exist_ok=True)
    
    def start(self, blocking: bool = True):
        """
        Start watching the folder
        
        Args:
            blocking: If True, blocks the main thread
        """
        self.handler = DataFileHandler(self.callback)
        self.observer = Observer()
        self.observer.schedule(self.handler, self.watch_path, recursive=False)
        
        print(f"👁️  Starting folder watcher...")
        print(f"   Watching: {self.watch_path}")
        print(f"   Supported formats: {DataFileHandler.SUPPORTED_EXTENSIONS}")
        print(f"\n⏳ Waiting for new files... (Press Ctrl+C to stop)\n")
        
        self.observer.start()
        
        if blocking:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping watcher...")
                self.stop()
    
    def stop(self):
        """Stop watching the folder"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            print("✅ Watcher stopped")
    
    def process_existing_files(self):
        """
        Process any existing files in the watch directory
        
        Useful for processing files that were added before the watcher started
        """
        print(f"📂 Checking for existing files in {self.watch_path}...")
        
        existing_files = []
        for ext in DataFileHandler.SUPPORTED_EXTENSIONS:
            pattern = f"*{ext}"
            existing_files.extend(Path(self.watch_path).glob(pattern))
        
        if existing_files:
            print(f"   Found {len(existing_files)} existing file(s)")
            for file_path in existing_files:
                print(f"   - {file_path.name}")
                try:
                    self.callback(str(file_path))
                except Exception as e:
                    print(f"   ❌ Error: {e}")
        else:
            print("   No existing files found")


def demo_callback(file_path: str):
    """Demo callback function"""
    print(f"🔄 Processing: {file_path}")
    print(f"   File size: {os.path.getsize(file_path):,} bytes")
    time.sleep(2)  # Simulate processing
    print(f"✅ Done processing: {file_path}")


if __name__ == "__main__":
    # Test the watcher
    import sys
    
    watch_dir = "data/input"
    
    if len(sys.argv) > 1:
        watch_dir = sys.argv[1]
    
    watcher = FolderWatcher(watch_dir, demo_callback)
    
    # Process any existing files first
    watcher.process_existing_files()
    
    # Start watching (blocking)
    watcher.start(blocking=True)
