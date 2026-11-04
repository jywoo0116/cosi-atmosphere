"""Geometry checks only; no Cosima execution or job submission.

Run with the cosi-atmosphere Python environment: python validation/check_geometry.py
The notebook uses these same checks. Generated files live in temporary directories.
"""
import ast
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import random
import subprocess
import tempfile

import numpy as np
import pandas as pd
from pymsis import msis

from cosi_atmosphere.response.MassModels import MakeMassModels

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "7a93224"
PROFILE_COLUMNS = ["mass_density[kg/m3]", "N2[m-3]", "O2[m-3]", "O[m-3]",
                   "He[m-3]", "H[m-3]", "Ar[m-3]", "N[m-3]",
                   "anomalous_oxygen[m-3]", "NO[m-3]", "Temperature[k]"]


@contextmanager
def in_directory(path):
    # The existing writer writes atmosphere.geo to the working directory.
    import os
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def controlled_profile(path, top_km):
    """Offline controlled MSIS inputs; not a measured atmosphere reproduction."""
    alts = np.arange(int(top_km * 10) + 1) / 10
    values = msis.run(np.array(["2016-06-13T12:00"], dtype="datetime64[m]"),
                      [-107.38], [-5.66], alts, version=2.1,
                      f107s=[100.], f107as=[100.], aps=np.full((1, 7), 4.))
    values = np.squeeze(values)
    assert np.all(np.isfinite(values[:, 0])) and np.all(values[:, 0] > 0)
    data = {"altitude[km]": alts}
    for i, name in enumerate(PROFILE_COLUMNS):
        # Match AtmosphericProfile's convention for unavailable species.
        data[name] = np.where(np.isnan(values[:, i]), 0., values[:, i])
    pd.DataFrame(data).to_csv(path, sep="\t", index=False, float_format="%10.6e")
    return MakeMassModels(path)


def generate(model, watch_alt, directory, writer=None):
    directory.mkdir(parents=True)
    rng_state = random.getstate()
    try:
        random.seed(1729)  # Match visualization colors for a strict regression.
        with in_directory(directory), redirect_stdout(io.StringIO()):
            if writer is None:
                model.spherical_model(watch_alt)
            else:
                writer(model, watch_alt)
        return (directory / "atmosphere.geo").read_text()
    finally:
        random.setstate(rng_state)


def original_writer():
    """Read the untouched writer from Git, without creating a backup module."""
    source = subprocess.check_output(
        ["git", "show", BASELINE_COMMIT + ":cosi_atmosphere/response/MassModels.py"],
        cwd=ROOT, text=True)
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "MakeMassModels")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "spherical_model")
    scope = {"np": np, "random": random}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "git-baseline", "exec"), scope)
    return scope["spherical_model"]


def verify_geometry(text, model, watch_alt):
    properties = {}
    for line in text.splitlines():
        fields = line.split()
        if fields:
            properties.setdefault(fields[0], []).append(fields[1:])
    def single(name):
        assert len(properties[name]) == 1, name
        return properties[name][0]
    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-6)

    earth = 6.378e8
    top = model.height[-1]
    thickness = (top / (len(model.height) - 1)) * 1e5
    source = np.array(single("SurroundingSphere"), dtype=float)
    close(source, [earth + top * 1e5, 0, 0, 0, earth + top * 1e5])
    shells = len(model.height) - 1
    assert sum(line.startswith("Volume VolumeSlice_") for line in text.splitlines()) == shells
    assert np.all(np.isfinite(model.density)) and np.all(model.density > 0)
    omitted = 0
    arrays = dict(H=model.H_normed, He=model.He_normed, N=model.N_normed,
                  O=model.O_normed, Ar=model.Ar_normed)
    # Check every shell's boundaries, density and all five elemental counts.
    for i in range(shells):
        material = "MaterialSlice_%d_%d" % (i, i + 1)
        volume = "VolumeSlice_%d_%d" % (i, i + 1)
        shape = single(volume + ".Shape")
        assert shape[0] == "SPHERE"
        close(np.array(shape[1:], dtype=float), [earth + i * thickness, earth + (i + 1) * thickness])
        assert float(single(material + ".Density")[0]) == model.density[i]
        assert single(volume + ".Material") == [material]
        written = {element: int(count) for element, count in properties[material + ".ComponentByAtoms"]}
        assert written and all(count > 0 for count in written.values())
        expected = {element: int(counts[i]) for element, counts in arrays.items() if counts[i] > 0}
        assert written == expected
        omitted += sum(counts[i] == 0 for counts in arrays.values())
    idx = int(np.flatnonzero(model.height == watch_alt)[0])
    watch = single("TestSphere.Shape")
    assert watch[0] == "Sphere"
    close(np.array(watch[1:], dtype=float), [earth + watch_alt * 1e5, earth + watch_alt * 1e5 + thickness])
    assert single("TestSphere.Mother") == ["VolumeSlice_%d_%d" % (idx, idx + 1)]
    assert single("TestSphere.Material") == ["MaterialSlice_%d_%d" % (idx, idx + 1)]
    return dict(top_km=float(top), watch_km=watch_alt, shells=shells,
                source_radius_km=float(source[0] / 1e5),
                omitted_zero_components=int(omitted))


def verify_rejected_inputs(model, directory):
    original_height = model.height.copy()
    examples = [([0.], 0.), ([1., 2.], 1.), ([0., 1., 3.], 1.),
                ([0., 1., 1.], 1.), ([0., np.nan], 0.),
                ([0., 1., 2.], .5), ([0., 1., 2.], 2.)]
    directory.mkdir()
    target = directory / "atmosphere.geo"
    target.write_text("existing geometry must survive invalid input\n")
    try:
        for heights, watched in examples:
            model.height = np.array(heights)
            try:
                with in_directory(directory):
                    model.spherical_model(watched)
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted invalid grid/watch: %r, %r" % (heights, watched))
            assert target.read_text() == "existing geometry must survive invalid input\n"
    finally:
        model.height = original_height
    return len(examples)


def run_checks(work):
    work = Path(work)
    balloon = controlled_profile(work / "profile_200.dat", 200)
    orbital = controlled_profile(work / "profile_1000.dat", 1000)
    original = original_writer()
    old_balloon = generate(balloon, 33.5, work / "old_balloon", original)
    new_balloon = generate(balloon, 33.5, work / "new_balloon")
    assert new_balloon == old_balloon, "200 km geometry differs from the Git baseline"
    old_orbital = generate(orbital, 550., work / "old_orbital", original)
    old_zeros = sum(".ComponentByAtoms " in line and int(line.split()[-1]) == 0
                    for line in old_orbital.splitlines())
    assert old_zeros > 0, "Controlled profile did not reproduce the zero-count bug"
    new_orbital = generate(orbital, 550., work / "new_orbital")
    return dict(balloon=verify_geometry(new_balloon, balloon, 33.5),
                orbital=verify_geometry(new_orbital, orbital, 550.),
                balloon_matches_original=True,
                original_orbital_zero_components=old_zeros,
                rejected_invalid_inputs=verify_rejected_inputs(orbital, work / "invalid_inputs"),
                transport="Not run; these checks validate generated geometry, not Geant4 initialization or physics.")


if __name__ == "__main__":
    import json
    with tempfile.TemporaryDirectory(prefix="cosi-geometry-") as work:
        print(json.dumps(run_checks(work), indent=2))
