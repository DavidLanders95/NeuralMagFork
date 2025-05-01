# MIT License
#
# Copyright (c) 2022-2025 NeuralMag team
#
# This file is part of NeuralMag – a simulation package for inverse micromagnetics.
# Repository: https://gitlab.com/neuralmag/neuralmag
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
                f"[{self.__class__.__name__}] Generate {config.backend.name} core methods"
            )
            code = str(self._generate_code(*args))
            with open(code_file_path, "w") as f:
                f.write(code)

        # import code
        module_spec = importlib.util.spec_from_file_location("code", code_file_path)
        self._code = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self._code)
