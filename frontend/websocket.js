import { ClipRecorder } from "./camera.js";

const GLOBAL_PROFILE_ID = "global";

const state = {
  ws: null,
  sessionId: null,
  camera: null,
  profileId: GLOBAL_PROFILE_ID,
  parameters: {
    captureFps: 25,
    captureCountdownSeconds: 3,
    commandClipMinSeconds: 2,
    commandClipMaxSeconds: 5,
    phraseCatalog: [],
  },
  streaming: false,
  connectionGeneration: 0,
  phase: "idle",
};

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

async function createSession() {
  const response = await fetch("/api/sessions", { method: "POST" });
  if (!response.ok) throw new Error("session creation failed");
  const body = await response.json();
  sessionStorage.setItem("silentVisionSessionId", body.sessionId);
  return body.sessionId;
}

function wsUrl(sessionId) {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/ws/${sessionId}`;
}

async function connect(connectionGeneration) {
  state.sessionId = await createSession();
  if (connectionGeneration !== state.connectionGeneration) throw new Error("stale start cancelled");
  const socketGeneration = connectionGeneration;
  const socket = new WebSocket(wsUrl(state.sessionId));
  state.ws = socket;
  socket.binaryType = "arraybuffer";
  socket.onmessage = (event) => {
    if (state.ws !== socket || socketGeneration !== state.connectionGeneration) return;
    handleEvent(JSON.parse(event.data));
  };
  await new Promise((resolve, reject) => {
    socket.onclose = (event) => {
      if (state.ws !== socket || socketGeneration !== state.connectionGeneration) {
        reject(new Error("stale websocket closed"));
        return;
      }
      console.info("Silent Vision WebSocket closed", {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean,
        sessionId: state.sessionId,
      });
      state.ws = null;
      if (state.phase === "done") {
        setText("socketStatus", "closed");
        return;
      }
      setStoppedUiState();
      reject(new Error("websocket closed before stream finished"));
    };
    socket.onopen = () => {
      if (state.ws !== socket || socketGeneration !== state.connectionGeneration) {
        reject(new Error("stale websocket opened"));
        return;
      }
      console.info("Silent Vision WebSocket opened", { sessionId: state.sessionId });
      setText("socketStatus", "connected");
      resolve();
    };
    socket.onerror = (event) => {
      if (state.ws !== socket || socketGeneration !== state.connectionGeneration) return;
      console.error("Silent Vision WebSocket error", event);
      reject(new Error("websocket connection failed"));
    };
  });
}

function renderCalibrationPhraseOptions() {
  const language = document.getElementById("calibration-language").value;
  const phraseSelect = document.getElementById("calibration-phrase-id");
  const catalog = Array.isArray(state.parameters.phraseCatalog)
    ? state.parameters.phraseCatalog
    : [];
  const phrases = catalog.filter((phrase) => phrase.language === language && phrase.enabled);
  const selectedPhraseId = phraseSelect.value;

  phraseSelect.replaceChildren();
  for (const phrase of phrases) {
    const option = document.createElement("option");
    option.value = phrase.phraseId;
    option.textContent = phrase.text;
    phraseSelect.append(option);
  }
  const unknownOption = document.createElement("option");
  unknownOption.value = "UNKNOWN";
  unknownOption.textContent = "UNKNOWN";
  phraseSelect.append(unknownOption);
  phraseSelect.disabled = false;

  if ([...phraseSelect.options].some((option) => option.value === selectedPhraseId)) {
    phraseSelect.value = selectedPhraseId;
  }
  updateUnknownPhraseInput();
}

function updateUnknownPhraseInput() {
  const phraseSelect = document.getElementById("calibration-phrase-id");
  const unknownPhrase = document.getElementById("calibration-unknown-phrase");
  const isUnknown = phraseSelect.value === "UNKNOWN";
  unknownPhrase.disabled = !isUnknown;
  unknownPhrase.required = isUnknown;
  if (!isUnknown) unknownPhrase.value = "";
}

function handleEvent(event) {
  if (event.type === "session.ready") {
    state.parameters = event.parameters;
    renderCalibrationPhraseOptions();
  }
  if (event.type === "vision.result") setText("visionStatus", event.faceDetected ? "mouth detected" : "mouth reused");
  if (event.type === "clip.received") setText("bufferStatus", `${event.bytes} bytes`);
  if (event.type === "command.result") {
    setText("lipStatus", event.intent);
    setText("semanticStatus", event.accepted ? "accepted" : `rejected: ${event.reason}`);
    setText("candidateOutput", JSON.stringify(event, null, 2));
  }
  if (event.type === "calibration.started") {
    setText("bufferStatus", "recording calibration");
    setText("calibration-result", JSON.stringify(event, null, 2));
  }
  if (event.type === "calibration.saved") {
    setText("bufferStatus", `${event.frames} frames saved`);
    setText("lipStatus", "sample saved");
    setText("semanticStatus", "calibrated");
    setText("agentStatus", "done");
    setText("calibration-result", JSON.stringify(event, null, 2));
    setDoneUiState();
  }
  if (event.type === "calibration.error") {
    setText("visionStatus", `calibration failed: ${event.message}`);
    setText("calibration-result", JSON.stringify(event, null, 2));
    setStoppedUiState();
  }
  if (event.type === "agent.result") {
    setText("agentStatus", event.action);
    setText("resultOutput", JSON.stringify(event, null, 2));
    setDoneUiState();
  }
  if (event.type === "stream.stopped") setStoppedUiState();
  if (event.type === "stream.committed") setText("visionStatus", "submitted");
  if (event.type === "error") setText("visionStatus", `${event.code}: ${event.message}`);
  if (event.type === "error") console.warn("Silent Vision server error", event);
}

function resetUiForNewStream() {
  state.phase = "preparing";
  setText("cameraStatus", "starting");
  setText("socketStatus", "connecting");
  setText("visionStatus", "waiting");
  setText("bufferStatus", "waiting");
  setText("lipStatus", "waiting");
  setText("semanticStatus", "waiting");
  setText("agentStatus", "waiting");
  setText("candidateOutput", "");
  setText("resultOutput", "");
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function runCaptureCountdown(connectionGeneration) {
  const seconds = state.parameters.captureCountdownSeconds ?? 3;
  for (let remaining = seconds; remaining > 0; remaining -= 1) {
    if (connectionGeneration !== state.connectionGeneration) throw new Error("countdown cancelled");
    setText("cameraStatus", `get ready: ${remaining}`);
    await delay(1000);
  }
  if (connectionGeneration !== state.connectionGeneration) throw new Error("countdown cancelled");
}

function setStoppedUiState() {
  state.phase = "idle";
  state.streaming = false;
  setText("cameraStatus", "stopped");
  setText("socketStatus", "closed");
  setText("visionStatus", "stopped");
  setText("bufferStatus", "waiting");
  setText("lipStatus", "stopped");
  setText("semanticStatus", "stopped");
  setText("agentStatus", "stopped");
  setText("candidateOutput", "");
  setText("resultOutput", "");
  document.getElementById("startButton").disabled = false;
  document.getElementById("stopButton").disabled = true;
  document.getElementById("save-sample").disabled = false;
}

function setAnalyzingUiState() {
  state.phase = "analyzing";
  state.streaming = false;
  setText("cameraStatus", "recorded");
  setText("socketStatus", "connected");
  setText("visionStatus", "submitted");
  setText("lipStatus", "waiting");
  setText("semanticStatus", "waiting");
  setText("agentStatus", "waiting");
  document.getElementById("startButton").disabled = true;
  document.getElementById("stopButton").disabled = false;
  document.getElementById("save-sample").disabled = true;
}

async function recordAndSendClip(connectionGeneration) {
  state.camera = new ClipRecorder({
    video: document.getElementById("cameraPreview"),
    fps: state.parameters.captureFps || 25,
  });
  await state.camera.startPreview();
  await runCaptureCountdown(connectionGeneration);
  if (connectionGeneration !== state.connectionGeneration || !state.ws) return;
  state.ws.send(JSON.stringify({
    type: "clip.start",
    profileId: GLOBAL_PROFILE_ID,
    language: document.getElementById("recognition-language").value,
  }));
  state.phase = "recording";
  state.streaming = true;
  setText("cameraStatus", "recording");
  setText("bufferStatus", "recording clip");
  const durationMs = Math.round((state.parameters.commandClipMaxSeconds || 5) * 1000);
  const clipBlob = await state.camera.recordClip({ durationMs });
  if (connectionGeneration !== state.connectionGeneration || !state.ws) return;
  stopCamera();
  state.streaming = false;
  state.ws.send(clipBlob);
  setAnalyzingUiState();
}

async function recordAndSendCalibrationClip(connectionGeneration) {
  const language = document.getElementById("calibration-language").value;
  const phraseId = document.getElementById("calibration-phrase-id").value;
  const phrase = phraseId === "UNKNOWN"
    ? document.getElementById("calibration-unknown-phrase").value.trim()
    : "";
  state.camera = new ClipRecorder({
    video: document.getElementById("cameraPreview"),
    fps: state.parameters.captureFps || 25,
  });
  await state.camera.startPreview();
  await runCaptureCountdown(connectionGeneration);
  if (connectionGeneration !== state.connectionGeneration || !state.ws) return;
  state.ws.send(JSON.stringify({
    type: "calibration.start",
    profileId: GLOBAL_PROFILE_ID,
    language,
    phraseId,
    phrase,
    scope: "global",
  }));
  state.phase = "recording";
  state.streaming = true;
  setText("cameraStatus", "recording sample");
  setText("bufferStatus", "recording calibration");
  const durationMs = Math.round((state.parameters.commandClipMaxSeconds || 5) * 1000);
  const clipBlob = await state.camera.recordClip({ durationMs });
  if (connectionGeneration !== state.connectionGeneration || !state.ws) return;
  stopCamera();
  state.streaming = false;
  state.ws.send(clipBlob);
  setAnalyzingUiState();
  setText("lipStatus", "saving sample");
  setText("semanticStatus", "waiting");
}

function setDoneUiState() {
  state.phase = "done";
  state.streaming = false;
  stopCamera();
  setText("cameraStatus", "done");
  setText("agentStatus", "done");
  document.getElementById("startButton").disabled = false;
  document.getElementById("stopButton").disabled = true;
  document.getElementById("save-sample").disabled = false;
}

function stopCamera() {
  if (state.camera) state.camera.stop();
  state.camera = null;
}

function closeSocket() {
  if (!state.ws) return;
  if (state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "stream.stop" }));
    state.ws.close(1000, "client stopped stream");
  } else if (state.ws.readyState === WebSocket.CONNECTING) {
    state.ws.close();
  }
  state.ws = null;
}

function cleanupCurrentStream() {
  state.connectionGeneration += 1;
  state.streaming = false;
  stopCamera();
  closeSocket();
}

document.getElementById("startButton").addEventListener("click", async () => {
  cleanupCurrentStream();
  const connectionGeneration = state.connectionGeneration;
  resetUiForNewStream();
  document.getElementById("startButton").disabled = true;
  document.getElementById("stopButton").disabled = false;
  try {
    await connect(connectionGeneration);
    if (connectionGeneration !== state.connectionGeneration || !state.ws) return;
    await recordAndSendClip(connectionGeneration);
  } catch (error) {
    if (connectionGeneration !== state.connectionGeneration) return;
    console.error("Silent Vision start failed", error);
    cleanupCurrentStream();
    setStoppedUiState();
    setText("visionStatus", `start failed: ${error.message}`);
  }
});

document.getElementById("save-sample").addEventListener("click", async () => {
  const phraseSelect = document.getElementById("calibration-phrase-id");
  const unknownPhrase = document.getElementById("calibration-unknown-phrase");
  if (phraseSelect.value === "UNKNOWN" && !unknownPhrase.value.trim()) {
    unknownPhrase.reportValidity();
    return;
  }
  cleanupCurrentStream();
  const connectionGeneration = state.connectionGeneration;
  resetUiForNewStream();
  document.getElementById("startButton").disabled = true;
  document.getElementById("stopButton").disabled = false;
  document.getElementById("save-sample").disabled = true;
  try {
    await connect(connectionGeneration);
    if (connectionGeneration !== state.connectionGeneration || !state.ws) return;
    await recordAndSendCalibrationClip(connectionGeneration);
  } catch (error) {
    if (connectionGeneration !== state.connectionGeneration) return;
    console.error("Silent Vision calibration failed", error);
    cleanupCurrentStream();
    setStoppedUiState();
    setText("visionStatus", `calibration failed: ${error.message}`);
  }
});

document.getElementById("stopButton").addEventListener("click", () => {
  cleanupCurrentStream();
  setStoppedUiState();
});

document.getElementById("calibration-language").addEventListener("change", renderCalibrationPhraseOptions);
document.getElementById("calibration-phrase-id").addEventListener("change", updateUnknownPhraseInput);

document.getElementById("profile-id").textContent = `Profile: ${state.profileId}`;
