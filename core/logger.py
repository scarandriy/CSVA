import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

class EvaluationLogger:
    """Handles logging of evaluation results."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def log(self, entry: Dict[str, Any]) -> None:
        """Log an evaluation entry.
        
        Args:
            entry: Dictionary containing evaluation data
        """
        entry["timestamp"] = datetime.now().isoformat()
        
        if not self.log_file.exists():
            self.log_file.write_text("[]")
        
        try:
            existing = json.loads(self.log_file.read_text())
        except json.JSONDecodeError:
            existing = []
            
        existing.append(entry)
        self.log_file.write_text(json.dumps(existing, indent=2)) 