from dataclasses import dataclass
from pathlib import Path

import jinja2


@dataclass
class Job:
    notebook: str
    backend: str


jobs = []

for notebook in (Path(__file__).parent.parent.parent / "docs/examples").glob("*.ipynb"):
    if notebook.stem.endswith("_jax"):
        backed = ["jax"]
    elif notebook.stem.endswith("_torch"):
        backends = ["torch"]
    else:
        backends = ["jax", "torch"]

    for backend in backends:
        jobs.append(Job(notebook.name, backend))

env = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))

template = env.get_template("notebook-pipeline.yml.j2")

with open("generated-notebook-pipeline.yml", "w") as f:
    f.write(template.render({"jobs": jobs}))
