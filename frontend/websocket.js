import { CameraStreamer } from "./camera.js";

const state = {
  ws: null,
  sessionId: null,
  camera: null,
  parameters: { captureFps: 25, windowFrames: 75, captureCountdownSeconds: 3 },
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

function handleEvent(event) {
  if (event.type === "session.ready") state.parameters = event.parameters;
  if (event.type === "vision.result") {
    const status = event.faceDetected ? "mouth detected" : "mouth reused";
    setText("visionStatus", status);
  }
  if (event.type === "buffer.progress") {
    setText("bufferStatus", `${event.bufferedFrames} / ${event.requiredFrames}`);
    autoSubmitWhenBufferFull(event);
  }
  if (event.type === "lip.candidates") {
    setText("lipStatus", "candidates ready");
    setText("semanticStatus", "running");
    setText("candidateOutput", JSON.stringify(event.candidates, null, 2));
  }
  if (event.type === "semantic.result") setText("semanticStatus", `${event.language}: ${event.text}`);
  if (event.type === "inference.started") setText("lipStatus", "running");
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
  setText("bufferStatus", `0 / ${state.parameters.windowFrames || 75}`);
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
  setText("bufferStatus", `0 / ${state.parameters.windowFrames || 75}`);
  setText("lipStatus", "stopped");
  setText("semanticStatus", "stopped");
  setText("agentStatus", "stopped");
  setText("candidateOutput", "");
  setText("resultOutput", "");
  document.getElementById("startButton").disabled = false;
  document.getElementById("stopButton").disabled = true;
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
}

function setDoneUiState() {
  state.phase = "done";
  state.streaming = false;
  stopCamera();
  setText("cameraStatus", "done");
  setText("agentStatus", "done");
  document.getElementById("startButton").disabled = false;
  document.getElementById("stopButton").disabled = true;
}

function autoSubmitWhenBufferFull(event) {
  if (state.phase !== "recording") return;
  if (event.bufferedFrames < event.requiredFrames) return;
  stopCamera();
  state.streaming = false;
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "stream.commit" }));
  }
  setAnalyzingUiState();
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
    state.camera = new CameraStreamer({
      video: document.getElementById("cameraPreview"),
      canvas: document.getElementById("captureCanvas"),
      fps: state.parameters.captureFps || 25,
      onFrame: (blob) => {
        if (!state.streaming || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
        if (state.ws.bufferedAmount > 2_000_000) return;
        state.ws.send(blob);
      },
    });
    await state.camera.startPreview();
    await runCaptureCountdown(connectionGeneration);
    if (connectionGeneration !== state.connectionGeneration) return;
    state.ws.send(JSON.stringify({ type: "stream.start" }));
    state.phase = "recording";
    state.streaming = true;
    state.camera.startCapture();
    setText("cameraStatus", "recording");
  } catch (error) {
    if (connectionGeneration !== state.connectionGeneration) return;
    console.error("Silent Vision start failed", error);
    cleanupCurrentStream();
    setStoppedUiState();
    setText("visionStatus", `start failed: ${error.message}`);
  }
});

document.getElementById("stopButton").addEventListener("click", () => {
  cleanupCurrentStream();
  setStoppedUiState();
});
