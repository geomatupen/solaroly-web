# backend/pvrt/core/projects.py
"""
Project management for organizing datasets, models, and sessions.
Each project has a nested structure:
  - train/ (contains data/train, data/valid, and outputs/)
  - test/ (contains data/test and outputs/)
  - overlays/
  - colmap/
"""

from pathlib import Path
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Project(BaseModel):
    """Project metadata and configuration."""
    id: str = Field(..., description="Unique project ID (UUID)")
    name: str = Field(..., description="Human-readable project name")
    description: Optional[str] = Field(default="", description="Project description")
    root_path: str = Field(..., description="Absolute path to project root directory")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    modified_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    thumbnail_path: Optional[str] = Field(default=None, description="Path to project thumbnail")
    
    # --- Top-level directories ---
    def get_train_dir(self) -> Path:
        """Get project's train directory (contains data and outputs)."""
        return Path(self.root_path) / "train"
    
    def get_test_dir(self) -> Path:
        """Get project's test directory (contains data and outputs)."""
        return Path(self.root_path) / "test"
    
    def get_overlays_dir(self) -> Path:
        """Get project's overlays directory."""
        return Path(self.root_path) / "overlays"

    def get_colmap_dir(self) -> Path:
        """Get project's colmap working directory."""
        return Path(self.root_path) / "colmap"
    
    # --- Train subdirectories ---
    def get_train_data_dir(self) -> Path:
        """Get project's training data directory (train/valid subfolders)."""
        return self.get_train_dir() / "data"
    
    def get_train_outputs_dir(self) -> Path:
        """Get project's training outputs directory (model runs)."""
        return self.get_train_dir() / "outputs"
    
    # --- Test subdirectories ---
    def get_test_data_dir(self) -> Path:
        """Get project's test data directory (uploaded test images)."""
        return self.get_test_dir() / "data"
    
    def get_test_outputs_dir(self) -> Path:
        """Get project's test outputs directory (detection results)."""
        return self.get_test_dir() / "outputs"
    
    # --- Legacy aliases for backward compatibility ---
    def get_data_dir(self) -> Path:
        """Legacy: Get training data directory."""
        return self.get_train_data_dir()
    
    def get_media_dir(self) -> Path:
        """Legacy media path (kept for backward compatibility)."""
        return Path(self.root_path) / "media"
    
    def get_models_dir(self) -> Path:
        """Legacy alias for training outputs directory."""
        return self.get_train_outputs_dir()
    
    def get_output_dir(self) -> Path:
        """Legacy alias for training outputs directory."""
        return self.get_train_outputs_dir()
    
    def get_sessions_dir(self) -> Path:
        """Legacy alias for test outputs directory."""
        return self.get_test_outputs_dir()
    
    def ensure_dirs(self) -> None:
        """Create all necessary project directories."""
        # Create top-level folders
        self.get_train_dir().mkdir(parents=True, exist_ok=True)
        self.get_test_dir().mkdir(parents=True, exist_ok=True)
        self.get_overlays_dir().mkdir(parents=True, exist_ok=True)
        self.get_colmap_dir().mkdir(parents=True, exist_ok=True)
        
        # Create train subdirectories
        self.get_train_data_dir().mkdir(parents=True, exist_ok=True)
        (self.get_train_data_dir() / "train").mkdir(parents=True, exist_ok=True)
        (self.get_train_data_dir() / "valid").mkdir(parents=True, exist_ok=True)
        self.get_train_outputs_dir().mkdir(parents=True, exist_ok=True)
        
        # Create test subdirectories
        self.get_test_data_dir().mkdir(parents=True, exist_ok=True)
        self.get_test_outputs_dir().mkdir(parents=True, exist_ok=True)


class ProjectManager:
    """Manages project registry and operations."""
    
    def __init__(self, registry_path: Path):
        """
        Initialize project manager.
        
        Args:
            registry_path: Path to projects.json registry file
        """
        self.registry_path = Path(registry_path)
        self._projects: Dict[str, Project] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load projects from registry JSON."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for project_id, project_data in data.get("projects", {}).items():
                        self._projects[project_id] = Project(**project_data)
            except (json.JSONDecodeError, TypeError):
                self._projects = {}
        else:
            self._projects = {}
    
    def _save_registry(self) -> None:
        """Save projects to registry JSON."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "projects": {
                project_id: project.model_dump()
                for project_id, project in self._projects.items()
            }
        }
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_project(self, project: Project) -> Project:
        """
        Create a new project.
        
        Args:
            project: Project instance
            
        Returns:
            Created project
        """
        if project.id in self._projects:
            raise ValueError(f"Project {project.id} already exists")
        
        project.ensure_dirs()
        self._projects[project.id] = project
        self._save_registry()
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        return self._projects.get(project_id)
    
    def list_projects(self) -> List[Project]:
        """List all projects."""
        return list(self._projects.values())
    
    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Optional[Project]:
        """Update project metadata."""
        project = self._projects.get(project_id)
        if not project:
            return None
        
        # Update allowed fields
        for key, value in updates.items():
            if key in {"name", "description", "thumbnail_path"}:
                setattr(project, key, value)
        
        project.modified_at = datetime.utcnow().isoformat()
        self._save_registry()
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """Delete project from registry (does not delete files)."""
        if project_id in self._projects:
            del self._projects[project_id]
            self._save_registry()
            return True
        return False
    
    def project_exists(self, project_id: str) -> bool:
        """Check if project exists."""
        return project_id in self._projects
