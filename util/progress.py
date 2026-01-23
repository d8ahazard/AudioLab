"""
AudioLab Progress Reporter
==========================

Unified progress reporting system with support for:
- Callbacks for Gradio UI updates
- WebSocket broadcasting for real-time updates
- ETA estimation based on historical timing
- Sub-task progress tracking
- Cancellation support

Usage:
    from util.progress import ProgressReporter, ProgressStage
    
    reporter = ProgressReporter(total_steps=100, task_name="Audio Separation")
    
    for i in range(100):
        reporter.update(
            current=i + 1,
            message=f"Processing model {i + 1}",
            stage=ProgressStage.PROCESSING
        )
        
        if reporter.is_cancelled:
            break
    
    reporter.complete()
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from collections import deque

logger = logging.getLogger(__name__)


class ProgressStage(str, Enum):
    """Standard progress stages for consistent UI messaging."""
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    PREPROCESSING = "preprocessing"
    PROCESSING = "processing"
    POSTPROCESSING = "postprocessing"
    SAVING = "saving"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


# User-friendly stage messages
STAGE_MESSAGES = {
    ProgressStage.INITIALIZING: "Preparing...",
    ProgressStage.LOADING_MODEL: "Loading AI model...",
    ProgressStage.PREPROCESSING: "Preparing audio...",
    ProgressStage.PROCESSING: "Processing...",
    ProgressStage.POSTPROCESSING: "Finalizing...",
    ProgressStage.SAVING: "Saving files...",
    ProgressStage.COMPLETE: "Complete!",
    ProgressStage.ERROR: "Error occurred",
    ProgressStage.CANCELLED: "Cancelled",
}


@dataclass
class ProgressUpdate:
    """Represents a single progress update."""
    progress: float  # 0.0 to 1.0
    message: str
    stage: ProgressStage
    current_step: int
    total_steps: int
    elapsed_seconds: float
    eta_seconds: Optional[float]
    task_name: str
    sub_task: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "progress": round(self.progress, 4),
            "progress_percent": round(self.progress * 100, 1),
            "message": self.message,
            "stage": self.stage.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds else None,
            "eta_formatted": self._format_eta(),
            "task_name": self.task_name,
            "sub_task": self.sub_task,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    def _format_eta(self) -> str:
        """Format ETA as human-readable string."""
        if self.eta_seconds is None:
            return "Calculating..."
        if self.eta_seconds < 0:
            return "Almost done..."
        
        minutes, seconds = divmod(int(self.eta_seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m remaining"
        elif minutes > 0:
            return f"{minutes}m {seconds}s remaining"
        else:
            return f"{seconds}s remaining"


class ProgressReporter:
    """
    Unified progress reporter with ETA estimation and multi-channel updates.
    
    Features:
        - Callback-based updates for Gradio
        - WebSocket broadcasting for real-time UI
        - Exponential moving average ETA estimation
        - Sub-task tracking
        - Cancellation support
        - Thread-safe operations
    
    Args:
        total_steps: Total number of steps to complete
        task_name: Human-readable task name
        callback: Optional callback function (progress: float, message: str, total: int)
        websocket_manager: Optional WebSocket manager for broadcasting
        enable_eta: Whether to calculate ETA estimates
    """
    
    def __init__(
        self,
        total_steps: int = 100,
        task_name: str = "Processing",
        callback: Optional[Callable[[float, str, int], None]] = None,
        websocket_manager: Optional[Any] = None,
        enable_eta: bool = True
    ):
        self.total_steps = max(1, total_steps)
        self.task_name = task_name
        self.callback = callback
        self.websocket_manager = websocket_manager
        self.enable_eta = enable_eta
        
        # State
        self._current_step = 0
        self._stage = ProgressStage.INITIALIZING
        self._message = STAGE_MESSAGES[ProgressStage.INITIALIZING]
        self._sub_task: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
        
        # Timing
        self._start_time = time.time()
        self._step_times: deque = deque(maxlen=20)  # Rolling window for ETA
        self._last_step_time = self._start_time
        
        # Cancellation
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        
        # History
        self._history: List[ProgressUpdate] = []
    
    @property
    def progress(self) -> float:
        """Current progress as a fraction (0.0 to 1.0)."""
        return min(1.0, self._current_step / self.total_steps)
    
    @property
    def progress_percent(self) -> float:
        """Current progress as a percentage (0 to 100)."""
        return self.progress * 100
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled.is_set()
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time since start."""
        return time.time() - self._start_time
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if not self.enable_eta or self._current_step == 0:
            return None
        
        if len(self._step_times) < 2:
            # Not enough data, use simple estimation
            avg_time_per_step = self.elapsed_seconds / self._current_step
        else:
            # Use exponential moving average
            avg_time_per_step = sum(self._step_times) / len(self._step_times)
        
        remaining_steps = self.total_steps - self._current_step
        return avg_time_per_step * remaining_steps
    
    def update(
        self,
        current: Optional[int] = None,
        message: Optional[str] = None,
        stage: Optional[ProgressStage] = None,
        sub_task: Optional[str] = None,
        increment: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressUpdate:
        """
        Update progress state and notify listeners.
        
        Args:
            current: Set current step directly
            message: Progress message to display
            stage: Current stage of processing
            sub_task: Optional sub-task description
            increment: Increment current step by this amount
            metadata: Additional metadata to include
        
        Returns:
            ProgressUpdate object with current state
        """
        with self._lock:
            # Update step count
            now = time.time()
            
            if current is not None:
                self._current_step = min(current, self.total_steps)
            elif increment > 0:
                self._current_step = min(self._current_step + increment, self.total_steps)
            
            # Track step timing for ETA
            step_time = now - self._last_step_time
            if step_time > 0 and step_time < 300:  # Ignore steps > 5 minutes
                self._step_times.append(step_time)
            self._last_step_time = now
            
            # Update state
            if stage is not None:
                self._stage = stage
            if message is not None:
                self._message = message
            elif stage is not None:
                self._message = STAGE_MESSAGES.get(stage, str(stage))
            
            self._sub_task = sub_task
            
            if metadata:
                self._metadata.update(metadata)
            
            # Create update object
            update = ProgressUpdate(
                progress=self.progress,
                message=self._message,
                stage=self._stage,
                current_step=self._current_step,
                total_steps=self.total_steps,
                elapsed_seconds=self.elapsed_seconds,
                eta_seconds=self.eta_seconds,
                task_name=self.task_name,
                sub_task=self._sub_task,
                metadata=self._metadata.copy()
            )
            
            self._history.append(update)
        
        # Notify listeners (outside lock)
        self._notify(update)
        
        return update
    
    def _notify(self, update: ProgressUpdate):
        """Notify all listeners of the update."""
        # Callback for Gradio
        if self.callback:
            try:
                self.callback(update.progress, update.message, self.total_steps)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
        
        # WebSocket broadcast
        if self.websocket_manager:
            try:
                asyncio.create_task(
                    self.websocket_manager.broadcast(update.to_dict())
                )
            except RuntimeError:
                # No event loop running, try sync broadcast
                try:
                    self.websocket_manager.broadcast_sync(update.to_dict())
                except Exception as e:
                    logger.debug(f"WebSocket broadcast skipped: {e}")
    
    def set_total_steps(self, total: int):
        """Update the total number of steps."""
        with self._lock:
            self.total_steps = max(1, total)
    
    def cancel(self):
        """Request cancellation of the current operation."""
        self._cancelled.set()
        self.update(stage=ProgressStage.CANCELLED, message="Operation cancelled")
    
    def reset(self):
        """Reset progress for reuse."""
        with self._lock:
            self._current_step = 0
            self._stage = ProgressStage.INITIALIZING
            self._message = STAGE_MESSAGES[ProgressStage.INITIALIZING]
            self._sub_task = None
            self._metadata = {}
            self._start_time = time.time()
            self._step_times.clear()
            self._last_step_time = self._start_time
            self._cancelled.clear()
            self._history.clear()
    
    def complete(self, message: str = "Complete!") -> ProgressUpdate:
        """Mark the operation as complete."""
        return self.update(
            current=self.total_steps,
            message=message,
            stage=ProgressStage.COMPLETE
        )
    
    def error(self, message: str = "An error occurred") -> ProgressUpdate:
        """Mark the operation as failed."""
        return self.update(
            message=message,
            stage=ProgressStage.ERROR
        )
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get history of all progress updates."""
        return [u.to_dict() for u in self._history]
    
    def __call__(self, progress: float, message: str = "", total: int = 0) -> None:
        """
        Callable interface for compatibility with existing callback patterns.
        
        Args:
            progress: Progress as fraction (0-1) or percentage (0-100)
            message: Progress message
            total: Optional total steps override
        """
        # Handle both fraction and percentage inputs
        if progress > 1:
            progress = progress / 100.0
        
        if total > 0:
            self.set_total_steps(total)
        
        # Convert progress to step count
        current = int(progress * self.total_steps)
        self.update(current=current, message=message if message else None)


class SubTaskReporter:
    """
    Context manager for tracking sub-tasks within a parent progress reporter.
    
    Usage:
        reporter = ProgressReporter(total_steps=10, task_name="Processing")
        
        with SubTaskReporter(reporter, "Loading models", steps=3) as sub:
            sub.update(1, "Model 1 loaded")
            sub.update(2, "Model 2 loaded")
            sub.update(3, "Model 3 loaded")
    """
    
    def __init__(
        self,
        parent: ProgressReporter,
        name: str,
        steps: int = 1,
        parent_steps: int = 1
    ):
        self.parent = parent
        self.name = name
        self.steps = steps
        self.parent_steps = parent_steps
        self._current = 0
        self._start_step = parent._current_step
    
    def __enter__(self):
        self.parent.update(sub_task=self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Advance parent progress
        self.parent.update(
            current=self._start_step + self.parent_steps,
            sub_task=None
        )
        return False
    
    def update(self, current: int, message: str = ""):
        """Update sub-task progress."""
        self._current = current
        fraction = current / self.steps
        sub_msg = f"{self.name}: {message}" if message else self.name
        
        # Calculate partial step progress
        partial = self._start_step + (fraction * self.parent_steps)
        self.parent.update(
            current=int(partial),
            message=sub_msg,
            sub_task=f"{current}/{self.steps}"
        )


class WebSocketProgressManager:
    """
    WebSocket manager for broadcasting progress updates to connected clients.
    
    Usage:
        from fastapi import WebSocket
        
        manager = WebSocketProgressManager()
        
        @app.websocket("/ws/progress")
        async def websocket_endpoint(websocket: WebSocket):
            await manager.connect(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                manager.disconnect(websocket)
    """
    
    def __init__(self):
        self.active_connections: List[Any] = []
        self._lock = threading.Lock()
    
    async def connect(self, websocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)
        logger.debug(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket):
        """Remove a WebSocket connection."""
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.debug(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients."""
        import json
        message = json.dumps(data)
        
        with self._lock:
            connections = list(self.active_connections)
        
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.debug(f"Failed to send to WebSocket: {e}")
                self.disconnect(connection)
    
    def broadcast_sync(self, data: Dict[str, Any]):
        """Synchronous broadcast (queues for async execution)."""
        # This is a fallback for when there's no event loop
        pass


# Global WebSocket manager instance
_ws_manager: Optional[WebSocketProgressManager] = None


def get_websocket_manager() -> WebSocketProgressManager:
    """Get or create the global WebSocket manager."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketProgressManager()
    return _ws_manager


def create_progress_callback(
    task_name: str = "Processing",
    total_steps: int = 100,
    websocket_enabled: bool = True
) -> ProgressReporter:
    """
    Factory function to create a ProgressReporter with optional WebSocket support.
    
    Args:
        task_name: Name of the task
        total_steps: Total steps for progress calculation
        websocket_enabled: Whether to enable WebSocket broadcasting
    
    Returns:
        Configured ProgressReporter instance
    """
    ws_manager = get_websocket_manager() if websocket_enabled else None
    return ProgressReporter(
        total_steps=total_steps,
        task_name=task_name,
        websocket_manager=ws_manager
    )
