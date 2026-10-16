document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const recordBtn = document.getElementById("record-btn");
  const recordText = document.getElementById("record-text");
  const analyzeBtn = document.getElementById("analyze-btn");
  const analyzeBtnText = document.getElementById("analyze-btn-text");

  const sampleBonafideBtn = document.getElementById("sample-bonafide-btn");
  const sampleSpoofBtn = document.getElementById("sample-spoof-btn");

  const audioPreviewBox = document.getElementById("audio-preview-box");
  const audioPlayer = document.getElementById("audio-player");
  const previewFilename = document.getElementById("preview-filename");
  const previewFilesize = document.getElementById("preview-filesize");
  const clearFileBtn = document.getElementById("clear-file-btn");

  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const resultsContent = document.getElementById("results-content");

  const verdictBanner = document.getElementById("verdict-banner");
  const verdictIcon = document.getElementById("verdict-icon");
  const verdictLabel = document.getElementById("verdict-label");
  const verdictDesc = document.getElementById("verdict-desc");
  const verdictStatusTag = document.getElementById("verdict-status-tag");

  const bonafideScore = document.getElementById("bonafide-score");
  const thresholdVal = document.getElementById("threshold-val");
  const marginCard = document.getElementById("margin-card");
  const marginVal = document.getElementById("margin-val");
  const marginHint = document.getElementById("margin-hint");
  const spectrumIndicator = document.getElementById("spectrum-indicator");

  const latencyVal = document.getElementById("latency-val");
  const latencyBreakdown = document.getElementById("latency-breakdown");
  const deviceVal = document.getElementById("device-val");
  const deviceName = document.getElementById("device-name");

  let stagedFile = null;

  // Check health on load
  fetchHealth();

  function fetchHealth() {
    fetch("/health")
      .then((res) => res.json())
      .then((data) => {
        if (deviceName && data.device) {
          deviceName.textContent = `Active on ${data.device.toUpperCase()}`;
        }
        if (deviceVal && data.device) {
          deviceVal.textContent = data.device.toUpperCase();
        }
      })
      .catch(() => {
        if (deviceName) deviceName.textContent = "Server Offline";
      });
  }

  // File Upload Handlers
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
      }
    });
  }

  function handleFile(file) {
    stagedFile = file;
    if (previewFilename) previewFilename.textContent = file.name;
    if (previewFilesize) {
      previewFilesize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    }

    // Play preview
    if (audioPlayer) {
      const audioUrl = URL.createObjectURL(file);
      audioPlayer.src = audioUrl;
    }
    if (audioPreviewBox) audioPreviewBox.style.display = "block";

    // Enable Analyze Button
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
    }

    // Auto-run analysis for immediate feedback
    scoreAudioFile(file);
  }

  // Analyze Button click
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", () => {
      if (stagedFile) {
        scoreAudioFile(stagedFile);
      }
    });
  }

  // Clear / Remove Audio 'X' button
  if (clearFileBtn) {
    clearFileBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      resetAudioState();
    });
  }

  function resetAudioState() {
    stagedFile = null;
    if (fileInput) fileInput.value = "";
    if (audioPlayer) {
      audioPlayer.pause();
      audioPlayer.removeAttribute("src");
      audioPlayer.load();
    }
    if (audioPreviewBox) {
      audioPreviewBox.style.display = "none";
    }
    if (analyzeBtn) {
      analyzeBtn.disabled = true;
    }
    if (analyzeBtnText) {
      analyzeBtnText.textContent = "Analyze Voice";
    }
    showEmpty();
  }

  // Benchmark Sample Buttons (Secondary Developer Area)
  if (sampleBonafideBtn) {
    sampleBonafideBtn.addEventListener("click", () => loadSample("bonafide"));
  }
  if (sampleSpoofBtn) {
    sampleSpoofBtn.addEventListener("click", () => loadSample("spoof"));
  }

  async function loadSample(type) {
    showLoading();
    try {
      const res = await fetch(`/api/sample/${type}`);
      if (!res.ok) throw new Error("Could not load sample");
      const blob = await res.blob();
      const ext = blob.type && blob.type.includes("wav") ? "wav" : "flac";
      const filename =
        type === "bonafide" ? "benchmark_human.wav" : "benchmark_ai_deepfake.wav";
      const file = new File([blob], filename, {
        type: blob.type || `audio/${ext}`,
      });
      handleFile(file);
    } catch (err) {
      alert("Error fetching benchmark sample audio: " + err.message);
      showEmpty();
    }
  }

  // Microphone Recording
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;

  if (recordBtn) {
    recordBtn.addEventListener("click", async () => {
      if (!isRecording) {
        startRecording();
      } else {
        stopRecording();
      }
    });
  }

  async function startRecording() {
    try {
      // Disable aggressive browser DSP noise suppression / echo cancellation
      // which strips speech formants and introduces robotic phase artifacts
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });
      } catch (e) {
        // Fallback to default if hardware constraints fail
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          if (!audioChunks || audioChunks.length === 0) return;

          // The browser records WebM/Opus by default. Convert to valid RIFF 16-bit PCM WAV.
          const rawBlob = new Blob(audioChunks, {
            type: mediaRecorder.mimeType || "audio/webm",
          });
          const arrayBuffer = await rawBlob.arrayBuffer();

          const AudioContextClass =
            window.AudioContext || window.webkitAudioContext;
          const audioCtx = new AudioContextClass();
          const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

          const wavBlob = audioBufferToWavBlob(audioBuffer);
          const recordedFile = new File([wavBlob], "microphone_record.wav", {
            type: "audio/wav",
          });

          if (audioCtx.state !== "closed") {
            await audioCtx.close();
          }

          handleFile(recordedFile);
        } catch (err) {
          console.error("Recording WAV conversion error:", err);
          alert("Failed to process recorded audio: " + err.message);
        }
      };

      mediaRecorder.start();
      isRecording = true;
      if (recordBtn) recordBtn.classList.add("recording");
      if (recordText) recordText.textContent = "Stop Recording";
    } catch (err) {
      alert("Microphone access error: " + err.message);
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach((track) => track.stop());
      isRecording = false;
      if (recordBtn) recordBtn.classList.remove("recording");
      if (recordText) recordText.textContent = "Record Voice";
    }
  }

  // API Call to /score
  async function scoreAudioFile(file) {
    showLoading();

    if (analyzeBtn) analyzeBtn.disabled = true;
    if (analyzeBtnText) analyzeBtnText.textContent = "Analyzing...";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/score", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server scoring error");
      }

      const data = await response.json();
      displayResults(data);
    } catch (error) {
      alert("Error scoring audio: " + error.message);
      showEmpty();
    } finally {
      if (analyzeBtn) analyzeBtn.disabled = false;
      if (analyzeBtnText) analyzeBtnText.textContent = "Analyze Voice";
    }
  }

  function showLoading() {
    if (emptyState) emptyState.style.display = "none";
    if (resultsContent) resultsContent.style.display = "none";
    if (loadingState) loadingState.style.display = "flex";
  }

  function showEmpty() {
    if (loadingState) loadingState.style.display = "none";
    if (resultsContent) resultsContent.style.display = "none";
    if (emptyState) emptyState.style.display = "flex";
  }

  function displayResults(data) {
    if (loadingState) loadingState.style.display = "none";
    if (emptyState) emptyState.style.display = "none";
    if (resultsContent) resultsContent.style.display = "block";

    const isBonafide = data.is_bonafide;
    const score = data.bonafide_score;
    const threshold = data.threshold;
    const marginRatio =
      data.margin_ratio !== undefined
        ? data.margin_ratio
        : score / (threshold || 1e-9);
    const logMargin =
      data.log_margin !== undefined
        ? data.log_margin
        : Math.log10(score) - Math.log10(threshold || 1e-9);

    // Update Verdict Banner
    if (verdictBanner) {
      verdictBanner.className = `verdict-banner ${
        isBonafide ? "bonafide" : "spoof"
      }`;
    }
    if (verdictIcon) {
      verdictIcon.innerHTML = isBonafide ? "✓" : "⚠";
    }
    if (verdictLabel) {
      verdictLabel.textContent = isBonafide
        ? "HUMAN VOICE"
        : "AI VOICE DETECTED";
    }
    if (verdictDesc) {
      verdictDesc.textContent = isBonafide
        ? "Likely genuine human speech"
        : "Likely synthetic / spoof speech";
    }
    if (verdictStatusTag) {
      verdictStatusTag.textContent = isBonafide
        ? "ABOVE THRESHOLD"
        : "BELOW THRESHOLD";
    }

    // Format scientific notation
    const formatSci = (val) => {
      if (val === undefined || val === null) return "--";
      return Number(val).toExponential(3);
    };

    if (bonafideScore) bonafideScore.textContent = formatSci(score);
    if (thresholdVal) thresholdVal.textContent = formatSci(threshold);

    // Margin Card
    if (marginCard) {
      marginCard.className = `stat-card stat-highlight ${
        isBonafide ? "bonafide-margin" : "spoof-margin"
      }`;
    }
    if (marginVal) {
      if (isBonafide) {
        marginVal.textContent =
          marginRatio >= 1000
            ? `${Math.round(marginRatio).toLocaleString()}×`
            : `${Number(marginRatio).toFixed(2)}×`;
      } else {
        marginVal.textContent = `${Number(marginRatio).toExponential(2)}×`;
      }
    }
    if (marginHint) {
      marginHint.textContent = isBonafide
        ? `+${Number(logMargin).toFixed(2)} orders of magnitude above threshold`
        : `${Number(logMargin).toFixed(2)} orders of magnitude below threshold`;
    }

    // Spectrum Indicator Position (Map log_margin [-12, +12] to [5%, 95%])
    // 0 log_margin is exact center (50%)
    if (spectrumIndicator) {
      const clampedLog = Math.max(-12, Math.min(12, logMargin));
      const indicatorPct = 50 + (clampedLog / 12) * 45;
      spectrumIndicator.style.left = `${indicatorPct.toFixed(1)}%`;
    }

    // Telemetry & Latency Breakdown
    if (latencyVal) latencyVal.textContent = `${data.latency_ms} ms`;
    if (latencyBreakdown) {
      if (data.latency) {
        latencyBreakdown.textContent = `Pre: ${data.latency.preprocess_ms}ms | Backbone: ${data.latency.backbone_ms}ms | Head: ${data.latency.classifier_ms}ms`;
      } else {
        latencyBreakdown.textContent = `Total inference time`;
      }
    }
    if (deviceVal) {
      deviceVal.textContent = data.device ? data.device.toUpperCase() : "CUDA";
    }
  }

  /**
   * Converts an AudioBuffer into a standard 16-bit mono PCM RIFF WAV Blob.
   * This guarantees full compatibility with soundfile/libsndfile on the server.
   */
  function audioBufferToWavBlob(audioBuffer) {
    const numChannels = 1; // mono
    const sampleRate = audioBuffer.sampleRate;
    const numSamples = audioBuffer.length;

    // Downmix to mono if multi-channel
    let channelData;
    if (audioBuffer.numberOfChannels > 1) {
      const ch0 = audioBuffer.getChannelData(0);
      const ch1 = audioBuffer.getChannelData(1);
      channelData = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        channelData[i] = (ch0[i] + ch1[i]) / 2.0;
      }
    } else {
      channelData = audioBuffer.getChannelData(0);
    }

    const bytesPerSample = 2; // 16-bit
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numSamples * blockAlign;
    const headerSize = 44;
    const totalSize = headerSize + dataSize;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    function writeString(offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }

    // 1. "RIFF" chunk descriptor
    writeString(0, "RIFF");
    view.setUint32(4, totalSize - 8, true);
    writeString(8, "WAVE");

    // 2. "fmt " sub-chunk
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);          // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);           // AudioFormat (1 = PCM)
    view.setUint16(22, numChannels, true); // NumChannels
    view.setUint32(24, sampleRate, true);  // SampleRate
    view.setUint32(28, byteRate, true);    // ByteRate
    view.setUint16(32, blockAlign, true);  // BlockAlign
    view.setUint16(34, 16, true);          // BitsPerSample (16 bits)

    // 3. "data" sub-chunk
    writeString(36, "data");
    view.setUint32(40, dataSize, true);

    // 4. Write 16-bit signed PCM samples
    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      let s = Math.max(-1, Math.min(1, channelData[i]));
      const intSample = s < 0 ? s * 0x8000 : s * 0x7FFF;
      view.setInt16(offset, intSample, true);
      offset += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
  }
});
