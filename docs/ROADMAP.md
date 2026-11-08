# Status and roadmap

## Status

| Component | State |
|---|---|
| Africa TTS (6 layers, 110M) | released, v0.1 (200k distillation steps, chosen by listening against 150k and 175k) |
| Africa TTS Base (24 layers, the distillation teacher) | released for research, v0.1 (step 100k) |
| Python package and CLI | released |
| PyTorch engine (fp32, CPU and GPU default in package and server) | released; unquantised high-fidelity synthesis on CPU and GPU |
| sherpa-onnx export and bundles (mobile) | released, tested end to end in sherpa-onnx 2.0.1 |
| API server (`server/`: Docker, Modal) | released; live behind the [demo Space](https://huggingface.co/spaces/AfriSpeech/africa-tts-demo) |
| C front end (africa-g2p + espeak-ng) | released, parity-tested on Linux, JNI tested on a desktop JVM |
| Android | Kotlin wrapper written; NDK build and on-device test still to do |
| iOS | not started |

## Roadmap

Help is welcome on any of these; see [CONTRIBUTING.md](../CONTRIBUTING.md).

- **Experimental languages:** collect native-speaker reports on the 657 untrained languages;
  promote the ones that work, and prioritise data for the ones that nearly do.
- **Evaluation:** extend the [benchmark](BENCHMARKS.md) with speaker similarity and
  native-speaker listening tests.
- **Numbers:** spell out numerals per language before G2P (the models never saw digits).
- **Tone:** tone-marked G2P tables where they are missing (Fon, Ewe, Igbo, Ga, ...), then retrain.
- **More data, more languages:** grow the languages under 2 hours, add missing ones, and add
  non-religious and conversational speech.
- **Mobile:** an Android example app and NDK build of the front end with espeak-ng 1.51; iOS.
- **Packaging:** africa-g2p and africa-tts releases on PyPI, so no git pins are needed.
- **Next training run:** fold in G2P fixes, new data and evaluation results, then distill again.
