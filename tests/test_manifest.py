import json

from browser_control.manifest import RunManifest


def test_manifest_writes_reproducibility_metadata(tmp_path) -> None:
    manifest = RunManifest(
        run_name="unit-test",
        seed=23,
        config={"environment": "fake-click-test"},
    )

    output_path = manifest.write(tmp_path / "manifest.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["run_name"] == "unit-test"
    assert payload["seed"] == 23
    assert payload["config"]["environment"] == "fake-click-test"
    assert payload["python_version"]
    assert set(payload["packages"]) == {
        "openenv-core",
        "torch",
        "transformers",
        "trl",
        "vllm",
    }

