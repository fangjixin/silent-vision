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
