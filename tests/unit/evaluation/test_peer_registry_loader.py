"""Unit tests for PeerRegistryLoader."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.evaluation.peer_registry_loader import PeerRegistryLoader
from src.evaluation.peer_solution import PeerSolution


class TestPeerRegistryLoader:
    """Test suite validating PeerRegistryLoader YAML serialization and catalog operations."""

    _SAMPLE_NAME: str = "competitor_nb_1"
    _SAMPLE_AUTHOR: str = "alice"
    _SAMPLE_URL: str = "https://kaggle.com/code/alice/nb1"
    _SAMPLE_DATE: str = "2026-09-30"
    _SAMPLE_PATH: str = "documentation/sample_code/nb1.ipynb"
    _SAMPLE_LB: float = 0.42

    def test_load_registry_parses_solutions(self, tmp_path: Path) -> None:
        """Verify load_registry correctly parses YAML solutions list into PeerSolution list."""
        yaml_file = tmp_path / "peer_registry.yaml"
        payload = {
            "solutions": [
                {
                    "name": self._SAMPLE_NAME,
                    "author": self._SAMPLE_AUTHOR,
                    "kaggle_url": self._SAMPLE_URL,
                    "lb_score": self._SAMPLE_LB,
                    "capture_date": self._SAMPLE_DATE,
                    "local_path": self._SAMPLE_PATH,
                    "status": "pending",
                    "key_techniques": ["reflection"],
                }
            ]
        }
        yaml_file.write_text(yaml.dump(payload), encoding="utf-8")
        loader = PeerRegistryLoader(str(yaml_file))
        solutions = loader.load_registry()
        assert len(solutions) == 1
        assert solutions[0].notebook_name == self._SAMPLE_NAME
        assert solutions[0].author == self._SAMPLE_AUTHOR
        assert solutions[0].lb_score == self._SAMPLE_LB
        assert solutions[0].key_techniques == ["reflection"]

    def test_save_registry_writes_yaml(self, tmp_path: Path) -> None:
        """Verify save_registry writes solution list to disk as valid YAML."""
        yaml_file = tmp_path / "saved_registry.yaml"
        loader = PeerRegistryLoader(str(yaml_file))
        sol = PeerSolution(
            notebook_name=self._SAMPLE_NAME,
            kaggle_url=self._SAMPLE_URL,
            author=self._SAMPLE_AUTHOR,
            lb_score=self._SAMPLE_LB,
            capture_date=self._SAMPLE_DATE,
            local_path=self._SAMPLE_PATH,
            analysis_status="pending",
        )
        loader.save_registry([sol])
        assert yaml_file.exists()
        reloaded = loader.load_registry()
        assert len(reloaded) == 1
        assert reloaded[0].notebook_name == self._SAMPLE_NAME

    def test_add_solution_appends(self, tmp_path: Path) -> None:
        """Verify add_solution appends new entry to existing registry file."""
        yaml_file = tmp_path / "append_registry.yaml"
        loader = PeerRegistryLoader(str(yaml_file))
        sol1 = PeerSolution(
            notebook_name="nb_1",
            kaggle_url="url1",
            author="auth1",
            lb_score=0.3,
            capture_date="2026-09-01",
            local_path="nb1.ipynb",
        )
        sol2 = PeerSolution(
            notebook_name="nb_2",
            kaggle_url="url2",
            author="auth2",
            lb_score=0.35,
            capture_date="2026-09-02",
            local_path="nb2.ipynb",
        )
        loader.save_registry([sol1])
        loader.add_solution(sol2)
        solutions = loader.load_registry()
        assert len(solutions) == 2
        assert solutions[1].notebook_name == "nb_2"

    def test_update_status_modifies_entry(self, tmp_path: Path) -> None:
        """Verify update_status mutates analysis_status and cv_score in registry."""
        yaml_file = tmp_path / "update_registry.yaml"
        loader = PeerRegistryLoader(str(yaml_file))
        sol = PeerSolution(
            notebook_name=self._SAMPLE_NAME,
            kaggle_url=self._SAMPLE_URL,
            author=self._SAMPLE_AUTHOR,
            lb_score=self._SAMPLE_LB,
            capture_date=self._SAMPLE_DATE,
            local_path=self._SAMPLE_PATH,
            analysis_status="pending",
        )
        loader.save_registry([sol])
        loader.update_status(self._SAMPLE_NAME, "tested", cv_score=0.48)
        updated = loader.load_registry()
        assert len(updated) == 1
        assert updated[0].analysis_status == "tested"
        assert updated[0].cv_score == 0.48

    def test_load_registry_nonexistent_or_empty_file(self, tmp_path: Path) -> None:
        """Verify load_registry returns empty list when file does not exist or has non-dict."""
        missing = tmp_path / "missing.yaml"
        loader = PeerRegistryLoader(str(missing))
        assert loader.load_registry() == []

        invalid = tmp_path / "invalid.yaml"
        invalid.write_text("string_content", encoding="utf-8")
        loader_inv = PeerRegistryLoader(str(invalid))
        assert loader_inv.load_registry() == []

    def test_load_registry_malformed_solutions(self, tmp_path: Path) -> None:
        """Verify load_registry handles solutions not being a list or containing non-dicts."""
        p1 = tmp_path / "not_list.yaml"
        p1.write_text("solutions: not_a_list", encoding="utf-8")
        assert PeerRegistryLoader(str(p1)).load_registry() == []

        p2 = tmp_path / "non_dict_entry.yaml"
        p2.write_text("solutions:\n  - 42\n  - null", encoding="utf-8")
        assert PeerRegistryLoader(str(p2)).load_registry() == []

    def test_update_status_non_matching_notebook(self, tmp_path: Path) -> None:
        """Verify update_status does not alter registry when notebook_name is not found."""
        yaml_file = tmp_path / "unmatched.yaml"
        loader = PeerRegistryLoader(str(yaml_file))
        sol = PeerSolution(
            notebook_name=self._SAMPLE_NAME,
            kaggle_url=self._SAMPLE_URL,
            author=self._SAMPLE_AUTHOR,
            lb_score=self._SAMPLE_LB,
            capture_date=self._SAMPLE_DATE,
            local_path=self._SAMPLE_PATH,
        )
        loader.save_registry([sol])
        loader.update_status("nonexistent_notebook", "tested")
        loaded = loader.load_registry()
        assert loaded[0].analysis_status == "pending"
