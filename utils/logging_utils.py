import os
import logging
import sys
from datetime import datetime


def setup_logging(log_file: str = None, console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file or None for console-only logging
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    
    # Create file handler if log_file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Log initial message
        root_logger.info(f"Logging to {log_file}")
    
    # Log initial message
    root_logger.info(f"Logging started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    

class ProgressLogger:
    """
    A logger class for tracking progress through multiple steps.
    """
    
    def __init__(self, logger_name: str = __name__, total_steps: int = None):
        """
        Initialize the ProgressLogger.
        
        Args:
            logger_name: Name for the logger instance
            total_steps: Total number of steps to complete
        """
        self.logger = logging.getLogger(logger_name)
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        
    def start_step(self, step_name: str) -> None:
        """
        Log the start of a step.
        
        Args:
            step_name: Name of the step
        """
        self.current_step += 1
        
        if self.total_steps:
            self.logger.info(f"Step {self.current_step}/{self.total_steps}: {step_name} - Started")
        else:
            self.logger.info(f"Step {self.current_step}: {step_name} - Started")
            
        self.step_start_time = datetime.now()
        
    def complete_step(self, step_name: str, additional_info: str = None) -> None:
        """
        Log the completion of a step.
        
        Args:
            step_name: Name of the step
            additional_info: Additional information to log
        """
        elapsed = datetime.now() - self.step_start_time
        
        if additional_info:
            if self.total_steps:
                self.logger.info(f"Step {self.current_step}/{self.total_steps}: {step_name} - Completed in {elapsed.total_seconds():.2f}s - {additional_info}")
            else:
                self.logger.info(f"Step {self.current_step}: {step_name} - Completed in {elapsed.total_seconds():.2f}s - {additional_info}")
        else:
            if self.total_steps:
                self.logger.info(f"Step {self.current_step}/{self.total_steps}: {step_name} - Completed in {elapsed.total_seconds():.2f}s")
            else:
                self.logger.info(f"Step {self.current_step}: {step_name} - Completed in {elapsed.total_seconds():.2f}s")
                
    def log_error(self, step_name: str, error: Exception) -> None:
        """
        Log an error during a step.
        
        Args:
            step_name: Name of the step
            error: Exception that occurred
        """
        if self.total_steps:
            self.logger.error(f"Step {self.current_step}/{self.total_steps}: {step_name} - Error: {str(error)}")
        else:
            self.logger.error(f"Step {self.current_step}: {step_name} - Error: {str(error)}")
            
    def finish(self) -> None:
        """Log the completion of all steps."""
        total_elapsed = datetime.now() - self.start_time
        
        self.logger.info(f"All {self.current_step} steps completed in {total_elapsed.total_seconds():.2f}s")


class StatsLogger:
    """
    A logger class for tracking and logging statistics.
    """
    
    def __init__(self, logger_name: str = __name__):
        """
        Initialize the StatsLogger.
        
        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self.stats = {}
        
    def add_stat(self, name: str, value: float, format_str: str = '{:.4f}') -> None:
        """
        Add a statistic to track.
        
        Args:
            name: Name of the statistic
            value: Value of the statistic
            format_str: Format string for printing the value
        """
        self.stats[name] = {
            'value': value,
            'format': format_str
        }
        
    def add_stats_dict(self, stats_dict: dict, prefix: str = '', format_str: str = '{:.4f}') -> None:
        """
        Add multiple statistics from a dictionary.
        
        Args:
            stats_dict: Dictionary of statistics
            prefix: Prefix to add to each statistic name
            format_str: Format string for printing the values
        """
        for name, value in stats_dict.items():
            key = f"{prefix}{name}" if prefix else name
            self.add_stat(key, value, format_str)
            
    def log_stats(self, header: str = "Statistics") -> None:
        """
        Log all tracked statistics.
        
        Args:
            header: Header for the statistics section
        """
        if not self.stats:
            self.logger.info(f"{header}: No statistics to report")
            return
            
        self.logger.info(f"{header}:")
        
        for name, stat in sorted(self.stats.items()):
            value = stat['value']
            format_str = stat['format']
            
            try:
                formatted_value = format_str.format(value)
            except (ValueError, TypeError):
                formatted_value = str(value)
                
            self.logger.info(f"  {name}: {formatted_value}")
            
    def clear(self) -> None:
        """Clear all tracked statistics."""
        self.stats = {}


def log_system_info() -> None:
    """Log system information for debugging purposes."""
    import platform
    import sys
    import psutil
    
    logger = logging.getLogger(__name__)
    
    try:
        # System information
        logger.info("System Information:")
        logger.info(f"  OS: {platform.platform()}")
        logger.info(f"  Python: {platform.python_version()}")
        
        # CPU information
        logger.info(f"  CPU: {platform.processor()}")
        logger.info(f"  Logical CPUs: {psutil.cpu_count(logical=True)}")
        logger.info(f"  Physical CPUs: {psutil.cpu_count(logical=False)}")
        
        # Memory information
        mem = psutil.virtual_memory()
        logger.info(f"  Total Memory: {mem.total / (1024**3):.2f} GB")
        logger.info(f"  Available Memory: {mem.available / (1024**3):.2f} GB")
        
        # Disk information
        disk = psutil.disk_usage('/')
        logger.info(f"  Disk Total: {disk.total / (1024**3):.2f} GB")
        logger.info(f"  Disk Free: {disk.free / (1024**3):.2f} GB")
        
        # Python environment
        logger.info(f"  Python Path: {sys.executable}")
        
    except Exception as e:
        logger.warning(f"Error getting system information: {str(e)}")
