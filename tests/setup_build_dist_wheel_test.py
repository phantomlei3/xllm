# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib.util
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from setuptools import Distribution


ROOT = Path(__file__).resolve().parents[1]


def _load_setup() -> ModuleType:
    spec = importlib.util.spec_from_file_location("xllm_setup", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_wheel(skip_test: bool, tmp_path: Path) -> list[str]:
    module = _load_setup()
    command = module.BuildDistWheel(Distribution({"name": "xllm"}))
    commands: list[str] = []

    command.get_finalized_command = mock.Mock(
        return_value=SimpleNamespace(device=None, arch=None, tilelang_jobs=None)
    )
    command.run_command = commands.append

    env = {"SKIP_TEST": "1"} if skip_test else {}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(module, "get_base_dir", return_value=str(tmp_path)),
        mock.patch.object(module, "get_python_version", return_value="312"),
        mock.patch.object(module.bdist_wheel, "run"),
    ):
        command.run()

    return commands


def test_skip_test_omits_command(tmp_path: Path) -> None:
    assert _run_wheel(True, tmp_path) == ["build"]


def test_default_runs_test_command(tmp_path: Path) -> None:
    assert _run_wheel(False, tmp_path) == ["build", "test"]
