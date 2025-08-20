"""
MicService - detect a usable microphone device.

Usage:
    from mic_service import MicService
    ms = MicService()
    device_index, sample_rate = ms.select_device()
    # pass to KWD and STT configs
"""

import sounddevice as sd
import numpy as np

class MicService:
    def __init__(self, sample_rates=(16000, 48000), probe_secs=0.3):
        self.sample_rates = sample_rates
        self.probe_secs = probe_secs

    def list_input_devices(self):
        """Return all input-capable devices."""
        devs = sd.query_devices()
        return [
            {
                "index": i,
                "name": d["name"],
                "max_input_channels": d["max_input_channels"],
                "default_samplerate": d.get("default_samplerate"),
            }
            for i, d in enumerate(devs)
            if d.get("max_input_channels", 0) > 0
        ]

    def _to_dbfs(self, x: np.ndarray) -> float:
        if x.size == 0:
            return -120.0
        if x.dtype == np.int16:
            xf = x.astype(np.float32) / 32768.0
        else:
            xf = x.astype(np.float32)
        rms = float(np.sqrt(np.mean(xf ** 2) + 1e-12))
        return 20 * np.log10(rms + 1e-12)

    def _probe_device(self, idx: int, sr: int) -> float | None:
        """Try to open and record a short chunk. Return idle RMS dBFS or None."""
        frames = int(sr * self.probe_secs)
        try:
            with sd.InputStream(device=idx, samplerate=sr, channels=1, dtype="int16") as stream:
                data, _ = stream.read(frames)
                return self._to_dbfs(data[:, 0])
        except Exception:
            return None

    def select_device(self):
        """
        Pick the 'best' input device.
        Prefers PipeWire/Pulse virtual devices if available, else first working hw.
        Returns (device_index, sample_rate).
        """
        candidates = self.list_input_devices()
        if not candidates:
            raise RuntimeError("No input-capable devices found")

        # Prefer virtual devices for sharing
        for dev in candidates:
            if any(v in dev["name"].lower() for v in ("pulse", "pipewire", "default")):
                for sr in self.sample_rates:
                    if self._probe_device(dev["index"], sr) is not None:
                        return dev["index"], sr

        # Fallback: first physical device that opens
        for dev in candidates:
            for sr in self.sample_rates:
                if self._probe_device(dev["index"], sr) is not None:
                    return dev["index"], sr

        raise RuntimeError("No usable microphone device could be opened")
