"""Run benchmarks/speed.py on a dedicated Modal CPU container (clean, reproducible hardware).

    modal run benchmarks/speed_modal.py --cpu 4
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .apt_install("espeak-ng", "git")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))
    .pip_install("sherpa-onnx>=1.12", "soundfile")
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(str(ROOT / "africa_tts"), "/root/africa_tts", ignore=["__pycache__"])
    .add_local_file(str(ROOT / "benchmarks" / "speed.py"), "/root/speed.py")
)
app = modal.App("africa-tts-speed", image=image)


def _run(threads: int) -> str:
    import subprocess

    subprocess.run(["python", "/root/speed.py", "--threads", str(threads), "--out", "/tmp/speed.json"],
                   check=True)
    return open("/tmp/speed.json").read()


@app.function(cpu=4.0, memory=8192, timeout=3600)
def speed4() -> str:
    return _run(4)


@app.function(cpu=2.0, memory=8192, timeout=3600)
def speed2() -> str:
    return _run(2)


@app.local_entrypoint()
def main(cpu: int = 4):
    out = (speed4 if cpu == 4 else speed2).remote()
    Path(f"speed_{cpu}cpu.json").write_text(out)
    print(out)
