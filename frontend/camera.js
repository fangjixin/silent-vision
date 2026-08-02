export class CameraStreamer {
  constructor({ video, canvas, fps, onFrame }) {
    this.video = video;
    this.canvas = canvas;
    this.fps = fps;
    this.onFrame = onFrame;
    this.stream = null;
    this.timer = null;
  }

  async startPreview() {
    this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  startCapture() {
    if (this.timer) return;
    const intervalMs = Math.round(1000 / this.fps);
    this.timer = window.setInterval(() => this.capture(), intervalMs);
  }

  async start() {
    await this.startPreview();
    this.startCapture();
  }

  capture() {
    if (!this.video.videoWidth || !this.video.videoHeight) return;
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
    const context = this.canvas.getContext("2d");
    context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    this.canvas.toBlob((blob) => {
      if (blob) this.onFrame(blob);
    }, "image/jpeg", 0.75);
  }

  stop() {
    if (this.timer) window.clearInterval(this.timer);
    this.timer = null;
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    this.stream = null;
  }
}

export class ClipRecorder {
  constructor({ video, fps, mimeType = "video/webm;codecs=vp9" }) {
    this.video = video;
    this.fps = fps;
    this.mimeType = MediaRecorder.isTypeSupported(mimeType) ? mimeType : "video/webm";
    this.stream = null;
  }

  async startPreview() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { frameRate: { ideal: this.fps, max: this.fps } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  async recordClip({ durationMs }) {
    if (!this.stream) await this.startPreview();
    const chunks = [];
    const recorder = new MediaRecorder(this.stream, {
      mimeType: this.mimeType,
      videoBitsPerSecond: 2_500_000,
    });
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) chunks.push(event.data);
    };
    const stopped = new Promise((resolve, reject) => {
      recorder.onstop = resolve;
      recorder.onerror = () => reject(new Error("video recording failed"));
    });
    recorder.start();
    await new Promise((resolve) => window.setTimeout(resolve, durationMs));
    if (recorder.state !== "inactive") recorder.stop();
    await stopped;
    return new Blob(chunks, { type: this.mimeType });
  }

  stop() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    this.stream = null;
  }
}
