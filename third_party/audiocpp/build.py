"""Build audio.cpp's C API library (Pocket TTS only, CPU) into africa_tts/lib/ for the wheel.

    python third_party/audiocpp/build.py [--src DIR] [--native] [--jobs N]

Clones audio.cpp at the commit in COMMIT, applies the patches in this folder, and builds the
shared library `audiocpp` with the ggml CPU backend. By default the CPU kernels are built in
every ISA variant and chosen at run time (portable wheels); --native builds for this CPU only.
Needs git, CMake >= 3.21 and a C++20 compiler.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
URL = "https://github.com/0xShug0/audio.cpp.git"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPO / "build" / "audio.cpp")
    ap.add_argument("--native", action="store_true", help="build for this CPU only")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    commit = (HERE / "COMMIT").read_text().strip()

    if not (a.src / ".git").exists():
        a.src.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", URL, str(a.src)])
    run(["git", "fetch", "--quiet", "origin", commit], cwd=a.src)
    run(["git", "checkout", "--quiet", "--force", commit], cwd=a.src)
    run(["git", "clean", "-fdq", "-e", "build"], cwd=a.src)
    for patch in sorted(HERE.glob("*.patch")):
        run(["git", "apply", str(patch)], cwd=a.src)

    build = a.src / "build" / ("native" if a.native else "portable")
    portable = not a.native and sys.platform != "darwin"  # macOS: one ISA per wheel
    cfg = ["cmake", "-S", str(a.src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
           "-DAUDIOCPP_MODEL_SET=custom", "-DAUDIOCPP_MODELS=pocket_tts",
           "-DAUDIOCPP_BUILD_C_API=ON", "-DAUDIOCPP_DEPLOYMENT_BUILD=ON",
           "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF",
           "-DENGINE_ENABLE_CUDA=OFF", "-DENGINE_ENABLE_VULKAN=OFF", "-DENGINE_ENABLE_METAL=OFF",
           f"-DENGINE_ENABLE_NATIVE_CPU={'ON' if a.native else 'OFF'}",
           f"-DENGINE_ENABLE_CPU_ALL_VARIANTS={'ON' if portable else 'OFF'}"]
    if sys.platform == "darwin":
        cfg.append("-DENGINE_ENABLE_OPENMP=OFF")  # Apple clang ships no OpenMP runtime
    if shutil.which("ninja"):
        cfg += ["-G", "Ninja"]
    run(cfg)
    run(["cmake", "--build", str(build), "--config", "Release", "--parallel", str(a.jobs),
         "--target", "audiocpp"])

    out = REPO / "africa_tts" / "lib"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    # The names the loader looks for (Linux: sonames such as libggml.so.0), each copied once
    # with symlinks resolved, since wheels cannot hold symlinks.
    keep = {"win32": r"^(audiocpp|ggml[\w-]*)\.dll$",
            "darwin": r"^lib(audiocpp|ggml[\w-]*)(\.\d+)?\.(dylib|so)$"}.get(
        sys.platform, r"^(libaudiocpp\.so|lib(?!audiocpp)[\w-]+\.so\.\d+|libggml-cpu-[\w-]+\.so)$")
    found = []
    for f in sorted(build.rglob("*")):
        if re.match(keep, f.name) and f.name not in found and (f.is_file() or f.is_symlink()):
            shutil.copy2(f.resolve(), out / f.name)
            found.append(f.name)
    if not any(n.startswith(("libaudiocpp", "audiocpp")) for n in found):
        sys.exit(f"audiocpp library not found under {build}")
    print("copied into africa_tts/lib:", ", ".join(sorted(found)))


if __name__ == "__main__":
    main()
