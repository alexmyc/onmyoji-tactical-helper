# -*- coding: utf-8 -*-
"""Cython C-Extension Build Script for Onmyoji Tactical Helper.

This script compiles custom Python source modules into C-extension binary files (.pyd / .so)
using Cython to protect intellectual property and optimize runtime execution performance.
"""

import os
from typing import List

from Cython.Build import cythonize
from setuptools import setup

USER_FILES: List[str] = [
    "action_bar_detector.py",
    "ai_tactical_manager.py",
    "ai_tactical_ui.py",
    "auth_module.py",
    "battle_context.py",
    "battle_monitor.py",
    "capture_engine.py",
    "db_manager.py",
    "llm_engine.py",
    "logic.py",
    "overlay_module.py",
    "target_selector.py",
    "turn_detector.py",
    "ui_module.py",
    "mode_controller.py"
]

if __name__ == "__main__":
    setup(
        name="OnmyojiTacticalHelperCore",
        ext_modules=cythonize(
            USER_FILES,
            compiler_directives={'language_level': "3"}
        )
    )