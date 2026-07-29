import { CameraStreamer } from "./camera.js";

const state = {
  ws: null,
  sessionId: null,
  camera: null,
  parameters: { captureFps: 25, windowFrames: 75 },
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
  state.ws.onclose = () => setText("socketStatus", "closed");
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
  if (event.type === "vision.result") setText("visionStatus", event.faceDetected ? "mouth detected" : "not detected");
  if (event.type === "buffer.progress") setText("bufferStatus", `${event.bufferedFrames} / ${event.requiredFrames}`);
  if (event.type === "lip.candidates") {
    setText("lipStatus", "candidates ready");
    setText("candidateOutput", JSON.stringify(event.candidates, null, 2));
  }
  if (event.type === "semantic.result") setText("semanticStatus", `${event.language}: ${event.text}`);
  if (event.type === "agent.result") {
    setText("agentStatus", event.action);
    setText("resultOutput", JSON.stringify(event, null, 2));
  }
  if (event.type === "error") setText("visionStatus", `${event.code}: ${event.message}`);
}

document.getElementById("startButton").addEventListener("click", async () => {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) await connect();
  state.ws.send(JSON.stringify({ type: "stream.start" }));
  state.camera = new CameraStreamer({
    video: document.getElementById("cameraPreview"),
    canvas: document.getElementById("captureCanvas"),
    fps: state.parameters.captureFps || 25,
    onFrame: (blob) => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(blob);
    },
  });
  await state.camera.start();
  setText("cameraStatus", "streaming");
  document.getElementById("startButton").disabled = true;
  document.getElementById("stopButton").disabled = false;
});

document.getElementById("stopButton").addEventListener("click", () => {
  if (state.camera) state.camera.stop();
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "stream.stop" }));
  setText("cameraStatus", "stopped");
  document.getElementById("startButton").disabled = false;
  document.getElementById("stopButton").disabled = true;
});
