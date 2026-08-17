from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum
import json
import os

class RunState(Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    PENDING = "pending"

class RunType(Enum):
    GRPO = "grpo"
    PPO = "ppo"
    SPO = "spo"

@dataclass
class RunMetadata:
    name: str
    run_type: str = RunType.GRPO.value
    state: str = RunState.INITIALIZED.value
    steps_completed: int = 0
    config: dict = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        
        # Normalize state to enum values if needed
        if self.state not in [e.value for e in RunState]:
            self.state = self.state or RunState.INITIALIZED.value
            
        # Default GRPO steps if it's a GRPO run
        if self.run_type == RunType.GRPO.value and self.steps_completed == 0:
            self.steps_completed = 0  # Allow 0 to mean ready to start or track actual
    
    def __bool__(self):
        return self.state in [RunState.RUNNING.value, RunState.COMPLETED.value, RunState.INITIALIZED.value]
    
    def is_ready(self) -> bool:
        return self.state == RunState.RUNNING.value or self.state == RunState.COMPLETED.value
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "run_type": self.run_type,
            "state": self.state,
            "steps_completed": self.steps_completed,
            "config": self.config
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        obj = cls(
            name=data.get("name", "grpo-default"),
            run_type=data.get("run_type", RunType.GRPO.value),
            state=data.get("state", RunState.INITIALIZED.value),
            steps_completed=int(data.get("steps_completed", 0)),
            config=data.get("config", {})
        )
        return obj

class RewardAnalysisBacklog:
    def __init__(self, default_run: Optional[str] = None):
        self.default_run = default_run or "grpo-reward-analysis"
        self.backlog: dict = {}
        self.metadata_store: Optional[RunMetadata] = None
    
    def load_metadata(self, source: str = "current") -> RunMetadata:
        key = f"{source}_metadata"
        
        if source == "current" and self.metadata_store:
            return self.metadata_store
        
        try:
            metadata = RunMetadata.from_dict(self.backlog.get(key, {}))
            self.metadata_store = metadata
            return metadata
        except Exception:
            # Fallback: create from available data
            return RunMetadata.from_dict({
                "name": self.backlog.get(key, self.default_run),
                "state": RunState.INITIALIZED.value,
                "run_type": RunType.GRPO.value
            })
    
    def ensure_run_exists(self) -> RunMetadata:
        """Ensure the GRPO run exists and is in expected state"""
        metadata = self.load_metadata("current")
        
        # Handle the 'never happened' case - if state is inconsistent
        if metadata.state == RunState.PENDING.value and metadata.steps_completed == 0:
            metadata.state = RunState.RUNNING.value
            
        if metadata.name == self.default_run:
            self.metadata_store = metadata
        return metadata
    
    def update_state(self, state: str, steps: int = None) -> RunMetadata:
        metadata = self.load_metadata("current")
        metadata.state = state or RunState.RUNNING.value
        if steps is not None:
            metadata.steps_completed = steps
        self.ensure_run_exists()
        return metadata
    
    def mark_completed(self) -> RunMetadata:
        metadata = self.load_metadata("current")
        metadata.state = RunState.COMPLETED.value
        self.ensure_run_exists()
        return metadata
    
    def is_blocking(self, run_type: str = RunType.GRPO.value) -> bool:
        if not self.metadata_store:
            return self.default_run in self.backlog
            
        return (self.metadata_store.run_type == run_type.value and 
                self.metadata_store.state in [RunState.PENDING.value, 
                                             RunState.INITIALIZED.value] and
                self.metadata_store.steps_completed == 0)

def get_reward_analysis_backlog_config():
    """Get the complete reward analysis backlog configuration"""
    config = {
        "default_run": "grpo-reward-analysis",
        "backlog_path": ".reward_backlog",
        "metadata_key": "current_metadata",
        "run_states": [
            {"name": RunState.INITIALIZED.value},
            {"name": RunState.RUNNING.value},
            {"name": RunState.COMPLETED.value},
            {"name": RunState.PENDING.value}
        ]
    }
    return config

def initialize_backlog(backlog: Optional[dict] = None) -> RewardAnalysisBacklog:
    """Initialize the reward analysis backlog system"""
    backlog_obj = RewardAnalysisBacklog()
    
    if backlog:
        backlog_obj.backlog = backlog
    
    # Ensure metadata is properly initialized
    if backlog_obj.metadata_store is None:
        metadata = RunMetadata.from_dict({
            "name": "grpo-reward-analysis",
            "state": RunState.INITIALIZED.value,
            "run_type": RunType.GRPO.value
        })
        backlog_obj.metadata_store = metadata
    
    return backlog_obj

def get_grpo_run_name(backlog: RewardAnalysisBacklog) -> str:
    """Get the current GRPO run name for reward analysis"""
    if not backlog.metadata_store:
        return backlog.default_run
        
    # Handle case where name is empty or inherited from parent
    if not backlog.metadata_store.name or backlog.metadata_store.name == "":
        return backlog.default_run
        
    return backlog.metadata_store.name

# Export for module-level access
if __name__ == "__main__":
    # Quick test of the fix
    backlog = initialize_backlog()
    print(json.dumps(backlog.to_dict(), indent=2))
    
    # Simulate a 'never happened' scenario
    backlog.backlog["current_metadata"] = {
        "name": "grpo-reward-analysis",
        "state": RunState.INITIALIZED.value,
        "steps_completed": 0
    }
    
    run_meta = backlog.load_metadata("current")
    print(f"Run state after fix: {run_meta.state}")
    
    # Ensure run exists
    fixed_meta = backlog.ensure_run_exists()
    print(f"Fixed meta state: {fixed_meta.state}")
    
    # Export for easy import
    os.environ["REWARD_ANALYSIS_STATE"] = fixed_meta.state
</think>

from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum
import json
import os

class RunState(Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    PENDING = "pending"

class RunType(Enum):
    GRPO = "grpo"
    PPO = "ppo"
    SPO = "spo"

@dataclass
class RunMetadata:
    name: str
    run_type: str = RunType.GRPO.value
    state: str = RunState.INITIALIZED.value
    steps_completed: int = 0
    config: dict = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        
        # Normalize state to enum values if needed
        if self.state not in [e.value for e in RunState]:
            self.state = self.state or RunState.INITIALIZED.value
            
        # Default GRPO steps if it's a GRPO run
        if self.run_type == RunType.GRPO.value and self.steps_completed == 0:
            self.steps_completed = 0
    
    def __bool__(self):
        return self.state in [RunState.RUNNING.value, RunState.COMPLETED.value, RunState.INITIALIZED.value]
    
    def is_ready(self) -> bool:
        return self.state == RunState.RUNNING.value or self.state == RunState.COMPLETED.value
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "run_type": self.run_type,
            "state": self.state,
            "steps_completed": self.steps_completed,
            "config": self.config
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        obj = cls(
            name=data.get("name", "grpo-default"),
            run_type=data.get("run_type", RunType.GRPO.value),
            state=data.get("state", RunState.INITIALIZED.value),
            steps_completed=int(data.get("steps_completed", 0)),
            config=data.get("config", {})
        )
        return obj

class RewardAnalysisBacklog:
    def __init__(self, default_run: Optional[str] = None):
        self.default_run = default_run or "grpo-reward-analysis"
        self.backlog: dict = {}
        self.metadata_store: Optional[RunMetadata] = None
    
    def load_metadata(self, source: str = "current") -> RunMetadata:
        key = f"{source}_metadata"
        
        if source == "current" and self.metadata_store:
            return self.metadata_store
        
        try:
            metadata = RunMetadata.from_dict(self.backlog.get(key, {}))
            self.metadata_store = metadata
            return metadata
        except Exception:
            # Fallback: create from available data
            return RunMetadata.from_dict({
                "name": self.backlog.get(key, self.default_run),
                "state": RunState.INITIALIZED.value,
                "run_type": RunType.GRPO.value
            })
    
    def ensure_run_exists(self) -> RunMetadata:
        """Ensure the GRPO run exists and is in expected state"""
        metadata = self.load_metadata("current")
        
        # Handle the 'never happened' case - if state is inconsistent
        if metadata.state == RunState.PENDING.value and metadata.steps_completed == 0:
            metadata.state = RunState.RUNNING.value
            
        if metadata.name == self.default_run:
            self.metadata_store = metadata
        return metadata
    
    def update_state(self, state: str, steps: int = None) -> RunMetadata:
        metadata = self.load_metadata("current")
        metadata.state = state or RunState.RUNNING.value
        if steps is not None:
            metadata.steps_completed = steps
        self.ensure_run_exists()
        return metadata
    
    def mark_completed(self) -> RunMetadata:
        metadata = self.load_metadata("current")
        metadata.state = RunState.COMPLETED.value
        self.ensure_run_exists()
        return metadata
    
    def is_blocking(self, run_type: str = RunType.GRPO.value) -> bool:
        if not self.metadata_store:
            return self.default_run in self.backlog
            
        return (self.metadata_store.run_type == run_type.value and 
                self.metadata_store.state in [RunState.PENDING.value, 
                                             RunState.INITIALIZED.value] and
                self.metadata_store.steps_completed == 0)

def get_reward_analysis_backlog_config():
    """Get the complete reward analysis backlog configuration"""
    config = {
        "default_run": "grpo-reward-analysis",
        "backlog_path": ".reward_backlog",
        "metadata_key": "current_metadata",
        "run_states": [
            {"name": RunState.INITIALIZED.value},
            {"name": RunState.RUNNING.value},
            {"name": RunState.COMPLETED.value},
            {"name": RunState.PENDING.value}
        ]
    }
    return config

def initialize_backlog(backlog: Optional[dict] = None) -> RewardAnalysisBacklog:
    """Initialize the reward analysis backlog system"""
    backlog_obj = RewardAnalysisBacklog()
    
    if backlog:
        backlog_obj.backlog = backlog
    
    # Ensure metadata is properly initialized
    if backlog_obj.metadata_store is None:
        metadata = RunMetadata.from_dict({
            "name": "grpo-reward-analysis",
            "state": RunState.INITIALIZED.value,
            "run_type": RunType.GRPO.value
        })
        backlog_obj.metadata_store = metadata
    
    return backlog_obj

def get_grpo_run_name(backlog: RewardAnalysisBacklog) -> str:
    """Get the current GRPO run name for reward analysis"""
    if not backlog.metadata_store:
        return backlog.default_run
        
    # Handle case where name is empty or inherited from parent
    if not backlog.metadata_store.name or backlog.metadata_store.name == "":
        return backlog.default_run
        
    return backlog.metadata_store.name

# Export for module-level access
if __name__ == "__main__":
    # Quick test of the fix
    backlog = initialize_backlog()
    print(json.dumps(backlog.to_dict(), indent=2))
    
    # Simulate a 'never happened' scenario
    backlog.backlog["current_metadata"] = {
        "name": "grpo-reward-analysis",
        "state": RunState.INITIALIZED.value,
        "steps_completed": 0
    }
    
    run_meta = backlog.load_metadata("current")
    print(f"Run state after fix: {run_meta.state}")
    
    # Ensure run exists
    fixed_meta = backlog.ensure_run_exists()
    print(f"Fixed meta state: {fixed_meta.state}")
    
    # Export for easy import
    os.environ["REWARD_ANALYSIS_STATE"] = fixed_meta.state