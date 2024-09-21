"""
NeuralMag - A nodal finite-difference code for inverse micromagnetics

Copyright (c) 2024 NeuralMag team

This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser Python General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Lesser Python General Public License for more details.

You should have received a copy of the Lesser Python General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import hashlib
import importlib
import os
import pathlib
import pickle

from neuralmag.common import logging
from neuralmag.common.config import config


class CodeClass(object):
    def save_and_load_code(self, *args):
        # setup cache file name
        this_module = pathlib.Path(importlib.import_module(self.__module__).__file__)
        i = this_module.parent.parts[::-1].index("neuralmag")
        prefix = "_".join(
            (config.backend.name,) + this_module.parent.parts[-i:] + (this_module.stem,)
        )
        cache_file = f"{prefix}_{hashlib.md5(pickle.dumps(args)).hexdigest()}.py"
        cache_dir = os.getenv(
            "NM_CACHEDIR", pathlib.Path.home() / ".cache" / "neuralmag"
        )
        code_file_path = cache_dir / cache_file

        # generate code
        if not code_file_path.is_file():
            code_file_path.parent.mkdir(parents=True, exist_ok=True)
            # TODO check if _generate_code method exists
            logging.info_green(
                f"[{self.__class__.__name__}] Generate torch core methods"
            )
            code = str(self._generate_code(*args))
            with open(code_file_path, "w") as f:
                f.write(code)

        # import code
        module_spec = importlib.util.spec_from_file_location("code", code_file_path)
        self._code = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self._code)
