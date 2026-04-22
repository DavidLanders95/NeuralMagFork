# SPDX-License-Identifier: MIT

import time

import numpy as np

import neuralmag as nm

SOLVER_LABELS = {
    "llg": "LLG relax",
    "bb": "BB steepest descent",
}


def _to_float(value):
    return float(np.asarray(nm.config.backend.to_numpy(value)).reshape(()))


def compare_static_methods(
    build_state,
    methods=("llg", "bb"),
    llg_builder=None,
    llg_runner=None,
    bb_builder=None,
    bb_runner=None,
):
    llg_builder = llg_builder or (lambda state: nm.LLGSolver(state))
    llg_runner = llg_runner or (lambda solver: solver.relax())
    bb_builder = bb_builder or (lambda state: nm.EnergyMinimizer(state, tol=1e3, max_iter=2000))
    bb_runner = bb_runner or (lambda solver: solver.minimize())

    results = {}
    for method in methods:
        state = build_state()
        t_start = time.perf_counter()

        if method == "llg":
            solver = llg_builder(state)
            llg_runner(solver)
            iterations = None
        elif method == "bb":
            solver = bb_builder(state)
            bb_runner(solver)
            iterations = solver.n_iter
        else:
            raise ValueError(f"Unknown minimization method: {method}")

        results[method] = {
            "state": state,
            "runtime_s": time.perf_counter() - t_start,
            "iterations": iterations,
            "energy": _to_float(state.E) if hasattr(state, "E") else None,
        }

    return results


def print_static_method_summary(title, results):
    print(f"{title} method comparison")
    baseline_energy = results.get("llg", {}).get("energy")
    for method, values in results.items():
        line = f"- {SOLVER_LABELS[method]}: runtime = {values['runtime_s']:.2f}s"
        if values["energy"] is not None:
            line += f", E = {values['energy']:.6e} J"
        if baseline_energy is not None and method != "llg" and values["energy"] is not None:
            line += f", ΔE vs LLG = {values['energy'] - baseline_energy:.3e} J"
        if values["iterations"] is not None:
            line += f", BB iterations = {values['iterations']}"
        print(line)