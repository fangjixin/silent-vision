const { test, expect } = require("@playwright/test");

const phraseCatalog = [
  {
    phraseId: "zh_light_on_hello",
    text: "你好，请帮我打开灯",
    language: "zh",
    intent: "LIGHT_ON",
    enabled: true,
  },
  {
    phraseId: "en_light_on_hello",
    text: "Hello, please turn on the light.",
    language: "en",
    intent: "LIGHT_ON",
    enabled: true,
  },
];

async function installRecordingFakes(page, { readyDelayMs = null } = {}) {
  await page.addInitScript(({ catalog, readyDelayMs: delayMs }) => {
    window.sentWebSocketMessages = [];
    window.fetch = async () => new Response(
      JSON.stringify({ sessionId: "browser-test-session" }),
      { status: 200, headers: { "content-type": "application/json" } },
    );

    class FakeWebSocket {
      static OPEN = 1;

      constructor() {
        this.readyState = FakeWebSocket.OPEN;
        Promise.resolve().then(() => {
          this.onopen?.();
          const sendReady = () => this.onmessage?.({
            data: JSON.stringify({
              type: "session.ready",
              parameters: {
                captureFps: 25,
                captureCountdownSeconds: 0,
                commandClipMaxSeconds: 0.001,
                phraseCatalog: catalog,
              },
            }),
          });
          if (delayMs === null) sendReady();
          else window.setTimeout(sendReady, delayMs);
        });
      }

      send(message) {
        window.sentWebSocketMessages.push(message);
      }

      close() {
        this.readyState = 3;
      }
    }

    window.WebSocket = FakeWebSocket;
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      configurable: true,
      value: async () => new MediaStream(),
    });
    HTMLMediaElement.prototype.play = async () => {};
    window.MediaRecorder = class {
      static isTypeSupported() {
        return true;
      }

      constructor() {
        this.state = "inactive";
      }

      start() {
        this.state = "recording";
      }

      stop() {
        this.state = "inactive";
        this.onstop?.();
      }
    };
  }, { catalog: phraseCatalog, readyDelayMs });
}

async function phraseOptionValues(page) {
  return page.locator("#calibration-phrase-id option").evaluateAll((options) => (
    options.map((option) => option.value)
  ));
}

async function sentMessage(page, type) {
  return page.evaluate((messageType) => window.sentWebSocketMessages
    .map((message) => {
      try {
        return JSON.parse(message);
      } catch {
        return null;
      }
    })
    .filter((message) => message?.type === messageType)
    .at(-1), type);
}

test("home page exposes camera controls and status", async ({ page }) => {
  await page.goto("http://127.0.0.1:8000/");
  await expect(page.locator("#startButton")).toBeVisible();
  await expect(page.locator("#cameraStatus")).toContainText("idle");
  await expect(page.locator("#bufferStatus")).toContainText("waiting");
});

test("waits for the catalog before fresh-page calibration", async ({ page }) => {
  await installRecordingFakes(page, { readyDelayMs: 50 });
  await page.goto("http://127.0.0.1:8000/");

  await page.locator("#save-sample").click();
  await expect.poll(() => sentMessage(page, "calibration.start")).toEqual({
    type: "calibration.start",
    profileId: "global",
    language: "zh",
    phraseId: "zh_light_on_hello",
    phrase: "",
    scope: "global",
  });
});

test("records the selected language and catalog phrase", async ({ page }) => {
  await installRecordingFakes(page);
  await page.goto("http://127.0.0.1:8000/");

  await expect(page.locator("#recognition-language option")).toHaveCount(2);
  expect(await page.locator("#recognition-language option").evaluateAll((options) => (
    options.map((option) => option.value)
  ))).toEqual(["zh", "en"]);
  await expect(page.locator("#calibration-language")).toBeVisible();
  await expect(page.locator("#calibration-phrase-id")).toBeVisible();
  await expect(page.locator("#calibration-unknown-phrase")).toBeVisible();
  await expect(page.locator("#calibration-intent")).toHaveCount(0);

  await page.locator("#recognition-language").selectOption("en");
  await page.locator("#startButton").click();
  await page.waitForFunction(() => window.sentWebSocketMessages.some((message) => {
    try {
      return JSON.parse(message).type === "clip.start";
    } catch {
      return false;
    }
  }));
  await expect.poll(() => sentMessage(page, "clip.start")).toEqual({
    type: "clip.start",
    profileId: "global",
    language: "en",
  });

  await page.locator("#calibration-language").selectOption("en");
  await expect.poll(() => phraseOptionValues(page)).toEqual([
    "en_light_on_hello",
    "UNKNOWN",
  ]);
  await expect(page.locator("#calibration-unknown-phrase")).toBeDisabled();
  await expect(page.locator("#calibration-unknown-phrase")).not.toHaveAttribute("required", "");

  await page.locator("#calibration-phrase-id").selectOption("UNKNOWN");
  await expect(page.locator("#calibration-unknown-phrase")).toBeEnabled();
  await expect(page.locator("#calibration-unknown-phrase")).toHaveAttribute("required", "");
  await page.locator("#calibration-unknown-phrase").fill("What time is it?");

  await page.locator("#stopButton").click();
  await expect(page.locator("#save-sample")).toBeEnabled();
  await page.locator("#save-sample").click();
  await expect.poll(() => sentMessage(page, "calibration.start")).toEqual({
    type: "calibration.start",
    profileId: "global",
    language: "en",
    phraseId: "UNKNOWN",
    phrase: "What time is it?",
    scope: "global",
  });

  await page.locator("#calibration-language").selectOption("zh");
  await expect.poll(() => phraseOptionValues(page)).toEqual([
    "zh_light_on_hello",
    "UNKNOWN",
  ]);
  await page.locator("#calibration-phrase-id").selectOption("zh_light_on_hello");
  await expect(page.locator("#calibration-unknown-phrase")).toBeDisabled();

  await page.locator("#stopButton").click();
  await expect(page.locator("#save-sample")).toBeEnabled();
  await page.locator("#save-sample").click();
  await expect.poll(() => sentMessage(page, "calibration.start")).toEqual({
    type: "calibration.start",
    profileId: "global",
    language: "zh",
    phraseId: "zh_light_on_hello",
    phrase: "",
    scope: "global",
  });
});
