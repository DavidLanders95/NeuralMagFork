# SPDX-License-Identifier: MIT

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
