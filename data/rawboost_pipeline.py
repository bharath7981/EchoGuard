import os
import sys
import numpy as np

sys.path.append(os.path.dirname(__file__))
from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise


class RawBoostAugmentor:
    """
    Applies RawBoost Algorithm 4 (Series combination: Convolutive + Impulsive + Stationary noise)
    using the exact parameters identified by Tak et al. as optimal for the ASVspoof LA track.
    """

    def __init__(
        self,
        sr: int = 16000,
        N_f: int = 5,
        nBands: int = 5,
        minF: int = 20,
        maxF: int = 8000,
        minBW: int = 100,
        maxBW: int = 1000,
        minCoeff: int = 10,
        maxCoeff: int = 100,
        minG: int = 0,
        maxG: int = 0,
        minBiasLinNonLin: int = 5,
        maxBiasLinNonLin: int = 20,
        P: int = 10,
        g_sd: int = 2,
        SNRmin: int = 10,
        SNRmax: int = 40,
    ):
        self.sr = sr
        self.N_f = N_f
        self.nBands = nBands
        self.minF = minF
        self.maxF = maxF
        self.minBW = minBW
        self.maxBW = maxBW
        self.minCoeff = minCoeff
        self.maxCoeff = maxCoeff
        self.minG = minG
        self.maxG = maxG
        self.minBiasLinNonLin = minBiasLinNonLin
        self.maxBiasLinNonLin = maxBiasLinNonLin
        self.P = P
        self.g_sd = g_sd
        self.SNRmin = SNRmin
        self.SNRmax = SNRmax

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies Algorithm 4 (1 + 2 + 3 in series) to a 1D float32 raw waveform.
        """
        # Ensure 1D float64 for stable FIR filter convolution
        x_proc = np.array(x, dtype=np.float64)

        # 1. Linear & non-linear convolutive noise (Ln-C)
        x_proc = LnL_convolutive_noise(
            x_proc,
            self.N_f,
            self.nBands,
            self.minF,
            self.maxF,
            self.minBW,
            self.maxBW,
            self.minCoeff,
            self.maxCoeff,
            self.minG,
            self.maxG,
            self.minBiasLinNonLin,
            self.maxBiasLinNonLin,
            self.sr,
        )

        # 2. Impulsive signal-dependent additive noise (ISD)
        x_proc = ISD_additive_noise(x_proc, self.P, self.g_sd)

        # 3. Stationary signal-independent additive noise (SSI)
        x_proc = SSI_additive_noise(
            x_proc,
            self.SNRmin,
            self.SNRmax,
            self.nBands,
            self.minF,
            self.maxF,
            self.minBW,
            self.maxBW,
            self.minCoeff,
            self.maxCoeff,
            self.minG,
            self.maxG,
            self.sr,
        )

        return np.array(x_proc, dtype=np.float32)


if __name__ == "__main__":
    augmentor = RawBoostAugmentor()
    dummy_audio = np.random.randn(64600).astype(np.float32)
    augmented = augmentor(dummy_audio)
    print(f"RawBoost Augmentor Test Successful: input shape {dummy_audio.shape}, output shape {augmented.shape}, dtype {augmented.dtype}")
