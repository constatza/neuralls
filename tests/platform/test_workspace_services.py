"""Tests for WorkspaceFactory (configuration/services.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.models.workspace import AssignmentWorkspace
from neuralls.platform.storage.workspaces import WorkspaceFactory


class TestWorkspaceFactory:
    """Tests for WorkspaceFactory class."""

    @pytest.fixture
    def factory(self, tmp_path: Path) -> WorkspaceFactory:
        """Create a WorkspaceFactory with temp directories."""
        output_root = tmp_path / "output"
        processed_root = tmp_path / "processed"

        output_root.mkdir()
        processed_root.mkdir()

        return WorkspaceFactory(
            output_root=output_root,
            processed_root=processed_root,
        )

    def test_factory_initialization(self, tmp_path: Path):
        """Test factory initializes with correct paths."""
        output_root = tmp_path / "output"
        processed_root = tmp_path / "processed"

        factory = WorkspaceFactory(
            output_root=output_root,
            processed_root=processed_root,
        )

        assert factory.output_root == output_root
        assert factory.processed_root == processed_root

    def test_create_workspace(self, factory: WorkspaceFactory, tmp_path: Path):
        """Test creating a workspace."""
        dataset_id = "test-dataset"
        run_id = "test-model"

        workspace = factory.create(dataset_id, run_id)

        assert isinstance(workspace, AssignmentWorkspace)
        assert workspace.dataset_id == dataset_id
        assert workspace.run_id == run_id

    def test_workspace_root_dir_structure(self, factory: WorkspaceFactory):
        """Test workspace root_dir follows output_root/dataset_id/run_id structure."""
        dataset_id = "my-dataset"
        run_id = "my-model"

        workspace = factory.create(dataset_id, run_id)

        expected_root = factory.output_root / dataset_id / run_id
        assert workspace.root_dir == expected_root

    def test_workspace_data_dir_structure(self, factory: WorkspaceFactory):
        """Test workspace data_dir follows processed_root/dataset_id structure."""
        dataset_id = "my-dataset"
        run_id = "my-model"

        workspace = factory.create(dataset_id, run_id)

        expected_data = factory.processed_root / dataset_id
        assert workspace.data_dir == expected_data

    def test_workspace_directories_created(self, factory: WorkspaceFactory):
        """Test that workspace creates checkpoint, figures, predictions dirs."""
        dataset_id = "test-dataset"
        run_id = "test-model"

        workspace = factory.create(dataset_id, run_id)

        # All subdirectories should be created
        assert workspace.checkpoint_dir.exists()
        assert workspace.checkpoint_dir.is_dir()

        assert workspace.figures_dir.exists()
        assert workspace.figures_dir.is_dir()

        assert workspace.predictions_dir.exists()
        assert workspace.predictions_dir.is_dir()

        assert workspace.data_dir.exists()
        assert workspace.data_dir.is_dir()

    def test_workspace_subdirectory_paths(self, factory: WorkspaceFactory):
        """Test that workspace subdirectories have correct paths."""
        dataset_id = "test-dataset"
        run_id = "test-model"

        workspace = factory.create(dataset_id, run_id)

        expected_root = factory.output_root / dataset_id / run_id

        assert workspace.checkpoint_dir == expected_root / "checkpoints"
        assert workspace.figures_dir == expected_root / "figures"
        assert workspace.predictions_dir == expected_root / "predictions"

    def test_create_multiple_workspaces(self, factory: WorkspaceFactory):
        """Test creating multiple workspaces with same factory."""
        workspace1 = factory.create("dataset1", "model1")
        workspace2 = factory.create("dataset2", "model2")

        assert workspace1.dataset_id == "dataset1"
        assert workspace1.run_id == "model1"

        assert workspace2.dataset_id == "dataset2"
        assert workspace2.run_id == "model2"

        # Paths should be different
        assert workspace1.root_dir != workspace2.root_dir
        assert workspace1.data_dir != workspace2.data_dir

    def test_workspace_nested_dataset_structure(self, factory: WorkspaceFactory):
        """Test workspaces for same dataset but different models."""
        dataset_id = "shared-dataset"
        model1_id = "model-a"
        model2_id = "model-b"

        workspace1 = factory.create(dataset_id, model1_id)
        workspace2 = factory.create(dataset_id, model2_id)

        # Same data directory (same dataset)
        assert workspace1.data_dir == workspace2.data_dir

        # Different experiment roots (different models)
        assert workspace1.root_dir != workspace2.root_dir

        # Both under same dataset parent
        assert workspace1.root_dir.parent == workspace2.root_dir.parent
        assert workspace1.root_dir.parent.name == dataset_id

    def test_workspace_frozen(self, factory: WorkspaceFactory):
        """Test that AssignmentWorkspace is frozen (immutable)."""
        workspace = factory.create("test", "test")

        with pytest.raises(AttributeError):
            workspace.dataset_id = "new_dataset"

    def test_workspace_paths_are_path_objects(self, factory: WorkspaceFactory):
        """Test that all workspace paths are Path objects."""
        workspace = factory.create("test", "test")

        assert isinstance(workspace.root_dir, Path)
        assert isinstance(workspace.data_dir, Path)
        assert isinstance(workspace.checkpoint_dir, Path)
        assert isinstance(workspace.figures_dir, Path)
        assert isinstance(workspace.predictions_dir, Path)

    def test_create_with_special_characters_in_ids(self, factory: WorkspaceFactory):
        """Test creating workspace with special characters in IDs."""
        dataset_id = "test-dataset-alpha"
        run_id = "model-2024-12-24"

        workspace = factory.create(dataset_id, run_id)

        assert workspace.dataset_id == dataset_id
        assert workspace.run_id == run_id

        # Directories should be created successfully
        assert workspace.checkpoint_dir.exists()
        assert workspace.data_dir.exists()

    def test_workspace_idempotent_creation(self, factory: WorkspaceFactory):
        """Test creating same workspace twice is idempotent."""
        dataset_id = "test-dataset"
        run_id = "test-model"

        workspace1 = factory.create(dataset_id, run_id)
        workspace2 = factory.create(dataset_id, run_id)

        # Should create same paths
        assert workspace1.root_dir == workspace2.root_dir
        assert workspace1.data_dir == workspace2.data_dir

        # Directories should still exist
        assert workspace2.checkpoint_dir.exists()
