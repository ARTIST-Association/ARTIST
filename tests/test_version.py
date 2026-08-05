"""Tests for ARTIST package version handling."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError

import artist


def test_version_from_distribution_metadata(monkeypatch) -> None:
    """Use the version from the artist-csp distribution metadata."""
    expected_version = "1.2.3"
    requested_distributions: list[str] = []

    def fake_version(distribution_name: str) -> str:
        requested_distributions.append(distribution_name)
        return expected_version

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    try:
        reloaded_artist = importlib.reload(artist)

        assert requested_distributions == ["artist-csp"]
        assert reloaded_artist.__version__ == expected_version
    finally:
        monkeypatch.undo()
        importlib.reload(artist)


def test_version_fallback_when_distribution_is_not_installed(
    monkeypatch,
) -> None:
    """Use version 0.0.0 when artist-csp metadata is unavailable."""
    requested_distributions: list[str] = []

    def fake_version(distribution_name: str) -> str:
        requested_distributions.append(distribution_name)
        raise PackageNotFoundError(distribution_name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    try:
        reloaded_artist = importlib.reload(artist)

        assert requested_distributions == ["artist-csp"]
        assert reloaded_artist.__version__ == "0.0.0"
    finally:
        monkeypatch.undo()
        importlib.reload(artist)
