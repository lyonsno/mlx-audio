"""Local Raon SpeechChat conversation with retained microphone/playback audio."""

import argparse
import json
import queue
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

RATE = 24_000
FRAME = 1_920


class AudioExchange:
    """Keep model execution and disk I/O outside the PortAudio callback."""

    def __init__(self):
        self.inputs = queue.Queue()
        self.recordings = queue.Queue()
        self.outputs = deque()
        self.lock = threading.Lock()
        self.underflow_samples = 0
        self.captured_samples = 0
        self.played_samples = 0

    def push_audio(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Generated playback must be finite mono PCM")
        if audio.size:
            with self.lock:
                self.outputs.append(audio.copy())

    def callback(self, indata, outdata, frames, clock, status):
        source = indata[:, 0].copy()
        outdata.fill(0)
        filled = 0
        with self.lock:
            while filled < frames and self.outputs:
                chunk = self.outputs[0]
                count = min(frames - filled, len(chunk))
                outdata[filled : filled + count, 0] = chunk[:count]
                filled += count
                if count == len(chunk):
                    self.outputs.popleft()
                else:
                    self.outputs[0] = chunk[count:]
        self.underflow_samples += frames - filled
        self.captured_samples += frames
        self.played_samples += filled
        self.inputs.put(source)
        self.recordings.put(
            (
                np.column_stack((source, outdata[:, 0])),
                {
                    "input_adc_time": float(clock.inputBufferAdcTime),
                    "output_dac_time": float(clock.outputBufferDacTime),
                    "samples": frames,
                    "status": str(status),
                },
            )
        )


def write_status(path, **fields):
    payload = {"updated_at": time.time(), **fields}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


class PCMRecording:
    """Retain exact PCM immediately; export a playable WAV on close."""

    def __init__(self, path, channels):
        self.path = path
        self.channels = channels
        self.raw_path = path.with_suffix(".f32")

    def __enter__(self):
        self.file = self.raw_path.open("wb")
        self.raw_path.with_suffix(".json").write_text(
            json.dumps({"sample_rate": RATE, "channels": self.channels, "dtype": "<f4"})
            + "\n"
        )
        return self

    def write(self, audio):
        self.file.write(np.asarray(audio, dtype="<f4").tobytes())
        self.file.flush()

    def flush(self):
        self.file.flush()

    def __exit__(self, *args):
        from scipy.io import wavfile

        self.file.close()
        if self.raw_path.stat().st_size:
            audio = np.memmap(self.raw_path, dtype="<f4", mode="r")
            if self.channels > 1:
                audio = audio.reshape(-1, self.channels)
            wavfile.write(self.path, RATE, audio)
        else:
            wavfile.write(self.path, RATE, np.empty((0, self.channels), np.float32))


def record_audio(exchange, directory, errors, stop):
    try:
        with (
            PCMRecording(
                directory / "microphone-left-playback-right.wav", channels=2
            ) as recording,
            (directory / "audio-clock.jsonl").open("w") as log,
        ):
            while True:
                item = exchange.recordings.get()
                if item is None:
                    break
                audio, metadata = item
                recording.write(audio)
                recording.flush()
                log.write(json.dumps(metadata) + "\n")
                log.flush()
    except BaseException as exc:
        errors.append(f"recording: {type(exc).__name__}: {exc}")
        stop.set()
        exchange.inputs.put(None)
        print(f"\nRecording failed: {exc}", flush=True)


def converse(session, directory, input_device, output_device, status_path):
    import sounddevice as sd

    exchange = AudioExchange()
    stop = threading.Event()
    quit_requested = threading.Event()
    errors = []
    recorder = threading.Thread(
        target=record_audio, args=(exchange, directory, errors, stop)
    )
    recorder.start()

    def callback(*args):
        try:
            exchange.callback(*args)
        except BaseException as exc:
            errors.append(f"audio callback: {type(exc).__name__}: {exc}")
            stop.set()
            exchange.inputs.put(None)
            raise sd.CallbackAbort() from exc

    def stop_on_enter():
        try:
            command = input().strip().lower()
            if command in {"q", "quit", "exit"}:
                quit_requested.set()
        except EOFError:
            errors.append("terminal input closed")
        finally:
            stop.set()
            exchange.inputs.put(None)

    def stream_finished():
        if not stop.is_set():
            errors.append("audio stream stopped unexpectedly")
            stop.set()
            exchange.inputs.put(None)

    frames_done = 0
    durations = []
    pending = np.empty(0, dtype=np.float32)
    started = time.monotonic()
    try:
        with (
            PCMRecording(directory / "generated.wav", channels=1) as generated,
            (directory / "frames.jsonl").open("w") as log,
        ):
            with sd.Stream(
                device=(input_device, output_device),
                samplerate=RATE,
                blocksize=FRAME,
                channels=(1, 1),
                dtype="float32",
                callback=callback,
                finished_callback=stream_finished,
            ) as stream:
                write_status(
                    status_path,
                    state="listening",
                    microphone_open=True,
                    recording_dir=str(directory),
                    sample_rate=stream.samplerate,
                    device=list(stream.device),
                    latency=list(stream.latency),
                )
                print("\nMIC LIVE. Enter: stop | q then Enter: quit.\n", flush=True)
                threading.Thread(target=stop_on_enter, daemon=True).start()
                while not stop.is_set():
                    chunk = exchange.inputs.get()
                    if chunk is None:
                        break
                    pending = np.concatenate((pending, chunk))
                    while pending.size >= FRAME and not stop.is_set():
                        frame, pending = pending[:FRAME], pending[FRAME:]
                        step_start = time.perf_counter()
                        result = session.step(frame)
                        audio = np.asarray(result.audio, dtype=np.float32)
                        exchange.push_audio(audio)
                        generated.write(audio)
                        elapsed = time.perf_counter() - step_start
                        durations.append(elapsed)
                        frames_done += 1
                        log.write(
                            json.dumps(
                                {
                                    "frame": result.frame_index,
                                    "decoded_frame": result.decoded_frame_index,
                                    "step_seconds": elapsed,
                                    "pending_input_blocks": exchange.inputs.qsize(),
                                    "pending_input_samples": int(pending.size),
                                }
                            )
                            + "\n"
                        )
                        log.flush()
                        if frames_done % 25 == 0:
                            print(
                                f"{frames_done / 12.5:.0f}s audio | "
                                f"step {elapsed * 1000:.0f}ms | "
                                f"input queue {exchange.inputs.qsize()} blocks",
                                flush=True,
                            )
    except BaseException as exc:
        errors.append(f"conversation: {type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        exchange.recordings.put(None)
        recorder.join()
        write_status(
            directory / "status.json",
            state="failed" if errors else "stopped",
            errors=errors,
            elapsed_seconds=time.monotonic() - started,
            captured_samples=exchange.captured_samples,
            processed_samples=frames_done * FRAME,
            played_generated_samples=exchange.played_samples,
            playback_silence_fill_samples=exchange.underflow_samples,
            pending_input_samples=int(pending.size),
            queued_input_blocks=exchange.inputs.qsize(),
            frames=frames_done,
            quit_requested=quit_requested.is_set(),
            median_step_seconds=float(np.median(durations)) if durations else None,
        )
        print(f"\nMic closed. Recording: {directory}", flush=True)
    return errors, quit_requested.is_set()


def menu_command():
    while True:
        command = (
            input("\nEnter: new conversation | q then Enter: quit > ").strip().lower()
        )
        if command in {"q", "quit", "exit"}:
            return "quit"
        if not command:
            return "start"
        print(
            "Unrecognized command. Press Enter to start, or type q then Enter to quit.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-audio", type=Path, required=True)
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir / "status.json"
    phase = "load"
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import sounddevice as sd
        from scipy.io import wavfile

        from mlx_audio.sts.models.raon.tts import RaonDuplexModel

        print("RAON SpeechChat | thinker 8-bit | local MLX Metal", flush=True)
        print("Loading and warming up. Microphone is CLOSED.", flush=True)
        write_status(status_path, state="loading", model=args.model)
        mx.random.seed(17)
        model = RaonDuplexModel.from_pretrained(args.model)
        nn.quantize(model.thinker, group_size=64, bits=8)
        mx.eval(model.parameters())

        def new_session():
            return model.create_duplex_session(
                system_prompt="You are engaging in real-time conversation.",
                speak_first=False,
                do_sample=True,
                temperature=0.9,
                top_k=66,
                top_p=0.95,
            )

        phase = "warmup"
        write_status(status_path, state="warming", microphone_open=False)
        rate, warmup = wavfile.read(args.warmup_audio)
        if (
            warmup.ndim != 1
            or rate != RATE
            or warmup.size == 0
            or warmup.dtype != np.float32
        ):
            raise ValueError("Warmup audio must be nonempty float32 mono 24 kHz")
        padding = (-warmup.size) % FRAME
        warmup = np.pad(warmup, (0, padding + 50 * FRAME))
        rehearsal = new_session()
        for offset in range(0, len(warmup), FRAME):
            result = rehearsal.step(warmup[offset : offset + FRAME])
            mx.eval(result.audio)
        del rehearsal, result
        sd.check_input_settings(device=args.input_device, channels=1, samplerate=RATE)
        sd.check_output_settings(device=args.output_device, channels=1, samplerate=RATE)
        write_status(
            status_path,
            state="awaiting_operator",
            microphone_open=False,
            model=args.model,
            thinker_quantization={"bits": 8, "group_size": 64},
            devices=str(sd.query_devices()),
        )
        print(sd.query_devices(), flush=True)
        print("\nHeadphones recommended. Keep other voice apps quiet.", flush=True)
        number = 0
        while True:
            if menu_command() == "quit":
                break
            number += 1
            phase = "session-create"
            mx.random.seed(17)
            session = new_session()
            directory = args.output_dir / f"conversation-{number:03d}"
            directory.mkdir()
            phase = "conversation"
            errors, quit_requested = converse(
                session, directory, args.input_device, args.output_device, status_path
            )
            del session
            if errors:
                raise RuntimeError("; ".join(errors))
            if quit_requested:
                break
            write_status(status_path, state="awaiting_operator", microphone_open=False)
        write_status(status_path, state="closed", conversations=number)
    except BaseException as exc:
        write_status(
            status_path,
            state="failed",
            phase=phase,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
