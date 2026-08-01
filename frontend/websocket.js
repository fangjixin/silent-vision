import { CameraStreamer } from "./camera.js";

const state = {
  ws: null,
  sessionId: null,
  camera: null,
  parameters: { captureFps: 25, windowFrames: 75 },
  streaming: false,
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

async function connect() {
  state.sessionId = await createSession();
  state.ws = new WebSocket(wsUrl(state.sessionId));
  state.ws.binaryType = "arraybuffer";
  state.ws.onclose = () => {
    setText("socketStatus", "closed");
    state.streaming = false;
    document.getElementById("startButton").disabled = false;
    document.getElementById("stopButton").disabled = true;
  };
  state.ws.onmessage = (event) => handleEvent(JSON.parse(event.data));
  await new Promise((resolve, reject) => {
    state.ws.onopen = () => {
      setText("socketStatus", "connected");
      resolve();
    };
    state.ws.onerror = () => reject(new Error("websocket connection failed"));
  });
}

function handleEvent(event) {
  if (event.type === "session.ready") state.parameters = event.parameters;
  if (event.type === "vision.result") {
    const status = event.faceDetected ? "mouth detected" : "mouth reused";
    setText("visionStatus", status);
  }
  if (event.type === "buffer.progress") setText("bufferStatus", `${event.bufferedFrames} / ${event.requiredFrames}`);
  if (event.type === "lip.candidates") {
    setText("lipStatus", "candidates ready");
    setText("candidateOutput", JSON.stringify(event.candidates, null, 2));
  }
  if (event.type === "semantic.result") setText("semanticStatus", `${event.language}: ${event.text}`);
  if (event.type === "inference.started") setText("lipStatus", "running");
  if (event.type === "agent.result") {
    setText("agentStatus", event.action);
    setText("resultOutput", JSON.stringify(event, null, 2));
  }
  if (event.type === "error") setText("visionStatus", `${event.code}: ${event.message}`);
}

function resetUiForNewStream() {
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
  state.streaming = false;
  stopCamera();
  closeSocket();
}

document.getElementById("startButton").addEventListener("click", async () => {
  cleanupCurrentStream();
  resetUiForNewStream();
  await connect();
  state.ws.send(JSON.stringify({ type: "stream.start" }));
  state.streaming = true;
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
  await state.camera.start();
  setText("cameraStatus", "streaming");
  document.getElementById("startButton").disabled = true;
  document.getElementById("stopButton").disabled = false;
});

document.getElementById("stopButton").addEventListener("click", () => {
  cleanupCurrentStream();
  setText("cameraStatus", "stopped");
  setText("socketStatus", "closed");
  document.getElementById("startButton").disabled = false;
  document.getElementById("stopButton").disabled = true;
});
