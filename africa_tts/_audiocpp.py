"""ctypes binding to audio.cpp's C API (libaudiocpp), for the default CPU engine.

The shared library ships inside the platform wheel (africa_tts/lib/); AFRICA_TTS_AUDIOCPP_LIB
points at another build. Only the calls Africa TTS needs: load a GGUF, open a CPU session,
synthesise (offline, or streaming by pulling audio events).
"""

import ctypes as C
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

OK, NOT_AVAILABLE = 0, 7
_LIB = None


class AudioCppError(RuntimeError):
    pass


class _ModelConfig(C.Structure):
    _fields_ = [("family_hint", C.c_char_p), ("config_id", C.c_char_p),
                ("weight_id", C.c_char_p), ("model_spec_override", C.c_char_p)]


class _BackendConfig(C.Structure):
    _fields_ = [("backend", C.c_char_p), ("device", C.c_int), ("threads", C.c_int)]


def _lib_path() -> Path:
    env = os.environ.get("AFRICA_TTS_AUDIOCPP_LIB")
    if env:
        return Path(env)
    name = {"win32": "audiocpp.dll", "darwin": "libaudiocpp.dylib"}.get(sys.platform, "libaudiocpp.so")
    return Path(__file__).parent / "lib" / name


def available() -> bool:
    return _lib_path().exists()


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _lib_path()
    if not path.exists():
        raise AudioCppError(
            f"the audio.cpp engine is not included in this installation ({path} is missing): "
            "install a platform wheel of africa-tts, or use the PyTorch engine "
            '(pip install "africa-tts[torch]", AfricaTTS.load(backend="torch"))')
    L = C.CDLL(str(path))
    P, S, N, F = C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_float)
    sig = {
        "audiocpp_last_error": (S, []),
        "audiocpp_registry_create": (N, [S, C.POINTER(P)]),
        "audiocpp_registry_free": (None, [P]),
        "audiocpp_model_load": (N, [P, S, C.POINTER(_ModelConfig), P, C.POINTER(P)]),
        "audiocpp_model_free": (None, [P]),
        "audiocpp_session_create": (N, [P, S, S, C.POINTER(_BackendConfig), P, C.POINTER(P)]),
        "audiocpp_session_free": (None, [P]),
        "audiocpp_session_run": (N, [P, P, C.POINTER(P)]),
        "audiocpp_request_create": (P, []),
        "audiocpp_request_free": (None, [P]),
        "audiocpp_request_set_text": (N, [P, S, S]),
        "audiocpp_request_set_voice_audio": (N, [P, F, C.c_size_t, N, N]),
        "audiocpp_request_set_option": (N, [P, S, S]),
        "audiocpp_result_free": (None, [P]),
        "audiocpp_result_audio": (N, [P, C.POINTER(F), C.POINTER(C.c_size_t), C.POINTER(N),
                                      C.POINTER(N)]),
        "audiocpp_result_named_audio_count": (C.c_size_t, [P]),
        "audiocpp_result_named_audio": (N, [P, C.c_size_t, C.POINTER(S), C.POINTER(F),
                                            C.POINTER(C.c_size_t), C.POINTER(N), C.POINTER(N)]),
        "audiocpp_stream_start": (N, [P, P]),
        "audiocpp_stream_next_event": (N, [P, C.POINTER(P)]),
        "audiocpp_stream_finish": (N, [P, C.POINTER(P)]),
        "audiocpp_event_free": (None, [P]),
        "audiocpp_event_is_final": (N, [P]),
        "audiocpp_event_as_result": (P, [P]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(L, name)
        fn.restype, fn.argtypes = res, args
    _LIB = L
    return L


def _check(status: int, what: str) -> None:
    if status != OK:
        err = lib().audiocpp_last_error()
        raise AudioCppError(f"{what} failed ({status}): {err.decode() if err else 'unknown error'}")


def _as_array(samples, frames: int, ch: int) -> np.ndarray:
    a = np.ctypeslib.as_array(samples, shape=(frames * max(ch, 1),)).astype(np.float32)
    return a.reshape(-1, ch).mean(1) if ch > 1 else a


def _named_audio(result) -> np.ndarray:
    """Concatenated named audio outputs (Pocket TTS streaming puts each chunk there)."""
    L, parts = lib(), []
    for i in range(L.audiocpp_result_named_audio_count(result)):
        name, samples, frames = C.c_char_p(), C.POINTER(C.c_float)(), C.c_size_t()
        sr, ch = C.c_int(), C.c_int()
        _check(L.audiocpp_result_named_audio(result, i, C.byref(name), C.byref(samples),
                                             C.byref(frames), C.byref(sr), C.byref(ch)),
               "reading streamed audio")
        if frames.value:
            parts.append(_as_array(samples, frames.value, ch.value))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _audio(result) -> np.ndarray:
    L, samples, frames = lib(), C.POINTER(C.c_float)(), C.c_size_t()
    sr, ch = C.c_int(), C.c_int()
    st = L.audiocpp_result_audio(result, C.byref(samples), C.byref(frames), C.byref(sr), C.byref(ch))
    if st == NOT_AVAILABLE or frames.value == 0:
        return np.zeros(0, dtype=np.float32)
    _check(st, "reading audio")
    return _as_array(samples, frames.value, ch.value)


class Engine:
    """One audio.cpp Pocket TTS model with an offline and a streaming CPU session."""

    def __init__(self, gguf: Path, threads: int, temperature: float) -> None:
        L = lib()
        self.temperature = temperature
        self._reg, self._model = C.c_void_p(), C.c_void_p()
        _check(L.audiocpp_registry_create(None, C.byref(self._reg)), "audio.cpp registry")
        cfg = _ModelConfig(b"pocket_tts", None, None, None)
        _check(L.audiocpp_model_load(self._reg, str(gguf).encode(), C.byref(cfg), None,
                                     C.byref(self._model)), f"loading {gguf}")
        self._backend = _BackendConfig(b"cpu", 0, threads)
        self._sessions: dict[str, C.c_void_p] = {}

    def _session(self, mode: str) -> C.c_void_p:
        if mode not in self._sessions:
            s = C.c_void_p()
            _check(lib().audiocpp_session_create(self._model, b"tts", mode.encode(),
                                                 C.byref(self._backend), None, C.byref(s)),
                   f"audio.cpp {mode} session")
            self._sessions[mode] = s
        return self._sessions[mode]

    def _request(self, ipa: str, voice: tuple[np.ndarray, int]):
        L, req = lib(), lib().audiocpp_request_create()
        _check(L.audiocpp_request_set_text(req, ipa.encode(), None), "setting text")
        audio = np.ascontiguousarray(voice[0], dtype=np.float32)
        _check(L.audiocpp_request_set_voice_audio(
            req, audio.ctypes.data_as(C.POINTER(C.c_float)), len(audio), voice[1], 1), "setting voice")
        _check(L.audiocpp_request_set_option(req, b"temperature", str(self.temperature).encode()),
               "setting temperature")
        return req

    def generate(self, ipa: str, voice: tuple[np.ndarray, int]) -> np.ndarray:
        L, req, res = lib(), self._request(ipa, voice), C.c_void_p()
        try:
            _check(L.audiocpp_session_run(self._session("offline"), req, C.byref(res)), "synthesis")
            return _audio(res)
        finally:
            if res:
                L.audiocpp_result_free(res)
            L.audiocpp_request_free(req)

    def stream(self, ipa: str, voice: tuple[np.ndarray, int]) -> Iterator[np.ndarray]:
        L, req = lib(), self._request(ipa, voice)
        session = self._session("streaming")
        sent = 0
        try:
            _check(L.audiocpp_stream_start(session, req), "starting stream")
            while True:
                ev = C.c_void_p()
                _check(L.audiocpp_stream_next_event(session, C.byref(ev)), "streaming")
                if not ev:
                    break
                try:
                    r = L.audiocpp_event_as_result(ev)
                    chunk = _named_audio(r)
                    if not len(chunk):
                        chunk = _audio(r)
                    final = L.audiocpp_event_is_final(ev)
                finally:
                    L.audiocpp_event_free(ev)
                if len(chunk):
                    sent += len(chunk)
                    yield chunk
                if final:
                    break
            res = C.c_void_p()  # finish closes the stream; its audio is the whole utterance
            if L.audiocpp_stream_finish(session, C.byref(res)) == OK and res:
                whole = _audio(res)
                L.audiocpp_result_free(res)
                if len(whole) > sent:  # only what the events did not deliver
                    yield whole[sent:]
        finally:
            L.audiocpp_request_free(req)

    def close(self) -> None:
        L = lib()
        for s in self._sessions.values():
            L.audiocpp_session_free(s)
        self._sessions.clear()
        if self._model:
            L.audiocpp_model_free(self._model)
            self._model = C.c_void_p()
        if self._reg:
            L.audiocpp_registry_free(self._reg)
            self._reg = C.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass
