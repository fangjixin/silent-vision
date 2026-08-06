# Phrase Catalog Preload and English UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the active phrase catalog before recording so the Phrase selector is immediately usable, while keeping every fixed interface string in English.

**Architecture:** A focused FastAPI route exposes `app.state.phrase_catalog` through the existing catalog serializer. The browser fetches that route on module startup and tracks whether its catalog came from HTTP or the active WebSocket; a `session.ready` catalog permanently supersedes a pending HTTP result. One calibration-availability function derives the Phrase and **Save Sample** disabled states from catalog authority and stream phase.

**Tech Stack:** Python 3.11, FastAPI, browser ES modules, pytest, Playwright.

## Global Constraints

- `GET /api/phrases` returns the catalog already stored in `app.state.phrase_catalog`; it must not load or maintain a second catalog.
- Prototype mode exposes the repository catalog, and Torch mode exposes the loaded checkpoint catalog.
- Initial catalog loading must not create a session, open a WebSocket, request camera permission, or begin a countdown.
- `session.ready.parameters.phraseCatalog` is authoritative for the rest of the page lifetime after it arrives; late HTTP success or failure must be ignored.
- **Save Sample** is enabled only when catalog authority is `http` or `websocket` and the phase is neither `preparing`, `recording`, nor `analyzing`.
- A language with no registered phrase renders only `UNKNOWN`, enables the custom phrase input, and remains recordable.
- The recognition **Start** control remains independent of catalog loading and failure.
- Both `zh` options display `Chinese`; all other fixed UI copy remains English.
- Chinese phrase content remains unchanged because it is recognition data, not interface copy.
- The existing WebSocket and recording payload contracts remain unchanged.
- A page reload is the only HTTP catalog retry mechanism.

---

## File Structure

- Create `api/phrases.py`: serve the active phrase catalog without creating a session.
- Modify `backend/main.py`: register the phrase API router.
- Create `tests/test_phrase_api.py`: verify the endpoint shape, active-state authority, and lack of session creation.
- Modify `frontend/index.html`: use English language labels and deterministic loading markup.
- Modify `frontend/websocket.js`: load, validate, render, and prioritize catalog sources; centralize calibration availability.
- Modify `tests/test_websocket_flow.py`: assert the served HTML contains only English language labels and starts calibration disabled.
- Modify `tests/e2e/camera.spec.js`: model HTTP and WebSocket catalogs independently and cover loading, errors, language filtering, response ordering, and recording payloads.

---

### Task 1: Read-Only Active Phrase Catalog API

**Files:**
- Create: `api/phrases.py`
- Modify: `backend/main.py:10-39`
- Create: `tests/test_phrase_api.py`

**Interfaces:**
- Consumes: `request.app.state.phrase_catalog: PhraseCatalog`
- Consumes: `catalog_records(catalog: PhraseCatalog) -> list[dict[str, object]]`
- Produces: `GET /api/phrases -> {"phrases": list[dict[str, object]]}`

- [ ] **Step 1: Write the failing API tests**

Create `tests/test_phrase_api.py`:

```python
from fastapi.testclient import TestClient

from command.catalog import PhraseCatalog, catalog_records


def test_get_phrases_returns_the_active_catalog_without_creating_a_session(
    app, monkeypatch
):
    def fail_if_session_is_created():
        raise AssertionError("phrase lookup must not create a session")

    monkeypatch.setattr(
        app.state.session_manager,
        "create_pending_session",
        fail_if_session_is_created,
    )

    response = TestClient(app).get("/api/phrases")

    assert response.status_code == 200
    assert response.json() == {
        "phrases": catalog_records(app.state.phrase_catalog)
    }


def test_get_phrases_reads_the_current_app_catalog(app):
    active_catalog = PhraseCatalog.from_records(
        [
            {
                "phraseId": "en_only",
                "text": "Have you eaten?",
                "language": "en",
                "intent": "CHAT_OTHER",
                "enabled": True,
            }
        ]
    )
    app.state.phrase_catalog = active_catalog

    response = TestClient(app).get("/api/phrases")

    assert response.status_code == 200
    assert response.json() == {"phrases": catalog_records(active_catalog)}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/test_phrase_api.py
```

Expected: both tests fail because `GET /api/phrases` returns 404.

- [ ] **Step 3: Add the focused API router**

Create `api/phrases.py`:

```python
from fastapi import APIRouter, Request

from command.catalog import catalog_records

router = APIRouter()


@router.get("/api/phrases")
async def get_phrases(request: Request) -> dict[str, list[dict[str, object]]]:
    return {"phrases": catalog_records(request.app.state.phrase_catalog)}
```

In `backend/main.py`, import and register it alongside the existing routers:

```python
from api.phrases import router as phrase_router
from api.session import router as session_router
from api.websocket import router as websocket_router
```

```python
app.include_router(phrase_router)
app.include_router(session_router)
app.include_router(websocket_router)
```

- [ ] **Step 4: Run the focused tests and lint**

Run:

```bash
pytest -q tests/test_phrase_api.py
ruff check api/phrases.py backend/main.py tests/test_phrase_api.py
```

Expected: two tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the API task**

```bash
git add api/phrases.py backend/main.py tests/test_phrase_api.py
git commit -m "feat: expose active phrase catalog"
```

---

### Task 2: Preloaded Catalog, English Labels, and Calibration State Gate

**Files:**
- Modify: `frontend/index.html:19-57`
- Modify: `frontend/websocket.js:5-384`
- Modify: `tests/test_websocket_flow.py:317-336`
- Modify: `tests/e2e/camera.spec.js:1-206`

**Interfaces:**
- Consumes: `GET /api/phrases` from Task 1.
- Consumes: `session.ready.parameters.phraseCatalog` from the existing WebSocket contract.
- Produces: `state.catalogAuthority` with values `loading`, `http`, `websocket`, or `error`.
- Produces: `loadPhraseCatalog() -> Promise<void>` for page-start loading.
- Produces: `syncCalibrationAvailability() -> void` as the only **Save Sample** availability writer.
- Preserves: `calibration.start` payload `{type, profileId, language, phraseId, phrase, scope}`.

- [ ] **Step 1: Update the browser fake so HTTP and WebSocket catalogs are independent**

In `tests/e2e/camera.spec.js`, keep `phraseCatalog`, then add a second valid catalog used to detect stale HTTP overwrites:

```javascript
const staleHttpCatalog = [
  {
    phraseId: "zh_stale_http",
    text: "迟到的目录",
    language: "zh",
    intent: "CHAT_OTHER",
    enabled: true,
  },
];
```

Replace `installRecordingFakes` with a version that intercepts only session creation and records side effects:

```javascript
async function installRecordingFakes(page, {
  readyDelayMs = null,
  websocketCatalog = phraseCatalog,
} = {}) {
  await page.addInitScript(({ catalog, readyDelayMs: delayMs }) => {
    window.sentWebSocketMessages = [];
    window.recordingTestCounters = {
      sessionRequests: 0,
      websocketConnections: 0,
      cameraRequests: 0,
    };
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const rawUrl = typeof input === "string" ? input : input.url;
      const url = new URL(rawUrl, window.location.origin);
      if (url.pathname !== "/api/sessions") return nativeFetch(input, init);
      window.recordingTestCounters.sessionRequests += 1;
      return new Response(
        JSON.stringify({ sessionId: "browser-test-session" }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    };

    class FakeWebSocket {
      static OPEN = 1;

      constructor() {
        window.recordingTestCounters.websocketConnections += 1;
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
      value: async () => {
        window.recordingTestCounters.cameraRequests += 1;
        return new MediaStream();
      },
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
  }, { catalog: websocketCatalog, readyDelayMs });
}
```

Add helpers for normal, failed, and deferred phrase responses:

```javascript
async function routePhraseCatalog(page, { catalog = phraseCatalog, status = 200 } = {}) {
  await page.route("**/api/phrases", (route) => route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(status === 200 ? { phrases: catalog } : { error: "failed" }),
  }));
}

async function deferPhraseCatalog(page, { catalog = staleHttpCatalog, status = 200 } = {}) {
  let releaseRequest;
  const gate = new Promise((resolve) => {
    releaseRequest = resolve;
  });
  await page.route("**/api/phrases", async (route) => {
    await gate;
    await route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(status === 200 ? { phrases: catalog } : { error: "failed" }),
    });
  });
  return releaseRequest;
}

async function recordingCounters(page) {
  return page.evaluate(() => window.recordingTestCounters);
}
```

- [ ] **Step 2: Write failing served-page and initial-load tests**

Extend `test_frontend_serves_language_and_catalog_calibration_controls` in `tests/test_websocket_flow.py`:

```python
assert response.text.count('<option value="zh">Chinese</option>') == 2
assert '<option value="en">English</option>' in response.text
assert ">中文<" not in response.text
assert '<select id="calibration-phrase-id" disabled>' in response.text
assert '<button id="save-sample" type="button" disabled>' in response.text
assert "Loading phrases..." in response.text
```

Add these Playwright tests in `tests/e2e/camera.spec.js`:

```javascript
test("preloads phrases without starting a session, socket, or camera", async ({ page }) => {
  await routePhraseCatalog(page);
  await installRecordingFakes(page);
  await page.goto("http://127.0.0.1:8000/");

  await expect(page.locator("#recognition-language option")).toHaveText(["Chinese", "English"]);
  await expect(page.locator("#calibration-language option")).toHaveText(["Chinese", "English"]);
  await expect.poll(() => phraseOptionValues(page)).toEqual([
    "zh_light_on_hello",
    "UNKNOWN",
  ]);
  await expect(page.locator("#save-sample")).toBeEnabled();
  await expect.poll(() => recordingCounters(page)).toEqual({
    sessionRequests: 0,
    websocketConnections: 0,
    cameraRequests: 0,
  });
});

test("keeps calibration disabled when phrase loading fails", async ({ page }) => {
  await routePhraseCatalog(page, { status: 503 });
  await installRecordingFakes(page);
  await page.goto("http://127.0.0.1:8000/");

  await expect(page.locator("#calibration-phrase-id")).toHaveText("Unable to load phrases");
  await expect(page.locator("#calibration-phrase-id")).toBeDisabled();
  await expect(page.locator("#save-sample")).toBeDisabled();
  await expect(page.locator("#startButton")).toBeEnabled();

  await page.locator("#stopButton").evaluate((button) => {
    button.disabled = false;
    button.click();
  });
  await expect(page.locator("#save-sample")).toBeDisabled();

  await page.locator("#startButton").click();
  await expect.poll(() => phraseOptionValues(page)).toEqual([
    "zh_light_on_hello",
    "UNKNOWN",
  ]);
  await page.locator("#stopButton").click();
  await expect(page.locator("#save-sample")).toBeEnabled();
});

test("allows UNKNOWN when the selected language has no registered phrase", async ({ page }) => {
  const englishOnlyCatalog = phraseCatalog.filter((phrase) => phrase.language === "en");
  await routePhraseCatalog(page, { catalog: englishOnlyCatalog });
  await installRecordingFakes(page, { websocketCatalog: englishOnlyCatalog });
  await page.goto("http://127.0.0.1:8000/");

  await expect.poll(() => phraseOptionValues(page)).toEqual(["UNKNOWN"]);
  await expect(page.locator("#calibration-unknown-phrase")).toBeEnabled();
  await expect(page.locator("#save-sample")).toBeEnabled();
});
```

- [ ] **Step 3: Write the failing HTTP/WebSocket precedence test**

Add this Playwright test:

```javascript
for (const lateStatus of [200, 503]) {
  test(`ignores a late HTTP ${lateStatus} after session.ready`, async ({ page }) => {
    const releaseHttp = await deferPhraseCatalog(page, {
      catalog: staleHttpCatalog,
      status: lateStatus,
    });
    await installRecordingFakes(page, { websocketCatalog: phraseCatalog });
    await page.goto("http://127.0.0.1:8000/");

    await expect(page.locator("#calibration-phrase-id")).toHaveText("Loading phrases...");
    await expect(page.locator("#save-sample")).toBeDisabled();

    await page.locator("#startButton").click();
    await expect.poll(() => phraseOptionValues(page)).toEqual([
      "zh_light_on_hello",
      "UNKNOWN",
    ]);

    const responseFinished = page.waitForResponse((response) => (
      response.url().endsWith("/api/phrases") && response.status() === lateStatus
    ));
    releaseHttp();
    await responseFinished;

    await expect.poll(() => phraseOptionValues(page)).toEqual([
      "zh_light_on_hello",
      "UNKNOWN",
    ]);
    await expect(page.locator("#calibration-phrase-id option").first()).not.toHaveValue("zh_stale_http");
  });
}
```

Update the existing recording tests to call `routePhraseCatalog(page)` before
`page.goto(...)`. Keep their existing assertions for `clip.start` and
`calibration.start`; they protect the unchanged payload contract.

- [ ] **Step 4: Run browser-facing tests and verify RED**

Run:

```bash
pytest -q tests/test_websocket_flow.py -k frontend
npm run test:e2e -- --grep "preloads|disabled|UNKNOWN|late HTTP"
```

Expected: the pytest assertion fails on `中文` and the missing disabled loading
markup; Playwright fails because no page-start phrase request exists and the old
state transitions enable **Save Sample** unconditionally.

- [ ] **Step 5: Add deterministic English loading markup**

In both language selectors in `frontend/index.html`, keep the protocol values and
change only the visible Chinese label:

```html
<option value="zh">Chinese</option>
<option value="en">English</option>
```

Give the Phrase selector and calibration button a safe server-rendered initial
state:

```html
<select id="calibration-phrase-id" disabled>
  <option selected>Loading phrases...</option>
</select>
```

```html
<button id="save-sample" type="button" disabled>Save Sample</button>
```

- [ ] **Step 6: Implement catalog validation, source precedence, and rendering**

Add `catalogAuthority` to the existing `state` object in
`frontend/websocket.js`:

```javascript
catalogAuthority: "loading",
```

Add these helpers after `setText`:

```javascript
function isValidPhraseCatalog(records) {
  return Array.isArray(records)
    && records.some((phrase) => phrase?.enabled === true)
    && records.every((phrase) => (
      phrase
      && typeof phrase.phraseId === "string"
      && phrase.phraseId.length > 0
      && typeof phrase.text === "string"
      && phrase.text.length > 0
      && ["zh", "en"].includes(phrase.language)
      && typeof phrase.intent === "string"
      && typeof phrase.enabled === "boolean"
    ));
}

function hasCatalogAuthority() {
  return ["http", "websocket"].includes(state.catalogAuthority);
}

function syncCalibrationAvailability() {
  const phraseSelect = document.getElementById("calibration-phrase-id");
  const saveSample = document.getElementById("save-sample");
  const busy = ["preparing", "recording", "analyzing"].includes(state.phase);
  const available = hasCatalogAuthority();
  phraseSelect.disabled = !available;
  saveSample.disabled = !available || busy;
  if (!available) {
    const unknownPhrase = document.getElementById("calibration-unknown-phrase");
    unknownPhrase.disabled = true;
    unknownPhrase.required = false;
  }
}

function showCatalogMessage(message) {
  const phraseSelect = document.getElementById("calibration-phrase-id");
  const option = document.createElement("option");
  option.textContent = message;
  option.selected = true;
  phraseSelect.replaceChildren(option);
  syncCalibrationAvailability();
}

function applyPhraseCatalog(records, authority) {
  if (!isValidPhraseCatalog(records)) throw new Error("invalid phrase catalog");
  if (authority === "http" && state.catalogAuthority === "websocket") return;
  state.parameters.phraseCatalog = records;
  state.catalogAuthority = authority;
  renderCalibrationPhraseOptions();
}

async function loadPhraseCatalog() {
  try {
    const response = await fetch("/api/phrases");
    if (!response.ok) throw new Error("phrase catalog request failed");
    const payload = await response.json();
    applyPhraseCatalog(payload.phrases, "http");
  } catch (error) {
    if (state.catalogAuthority === "websocket") return;
    console.error("Silent Vision phrase catalog failed", error);
    state.catalogAuthority = "error";
    showCatalogMessage("Unable to load phrases");
  }
}
```

Keep `renderCalibrationPhraseOptions()` catalog-driven, but remove its direct
`phraseSelect.disabled = false` assignment and finish it with the shared gate:

```javascript
if ([...phraseSelect.options].some((option) => option.value === selectedPhraseId)) {
  phraseSelect.value = selectedPhraseId;
}
updateUnknownPhraseInput();
syncCalibrationAvailability();
```

With no matching known phrase, the existing unconditional `UNKNOWN` append makes
it the selected option, and `updateUnknownPhraseInput()` enables the custom input.

Change the `session.ready` branch so runtime parameters and WebSocket authority
arrive together:

```javascript
if (event.type === "session.ready") {
  state.parameters = event.parameters;
  applyPhraseCatalog(event.parameters.phraseCatalog, "websocket");
}
```

At the bottom of the module, after registering event listeners, start only the
HTTP catalog request:

```javascript
document.getElementById("profile-id").textContent = `Profile: ${state.profileId}`;
syncCalibrationAvailability();
void loadPhraseCatalog();
```

- [ ] **Step 7: Make the shared gate authoritative across stream transitions**

In `resetUiForNewStream()`, `setStoppedUiState()`, `setAnalyzingUiState()`, and
`setDoneUiState()`, call `syncCalibrationAvailability()` after assigning
`state.phase`. Remove every direct assignment to
`document.getElementById("save-sample").disabled` from those functions and from
the two click handlers.

The resulting phase endings must have these exact shapes:

```javascript
function resetUiForNewStream() {
  state.phase = "preparing";
  syncCalibrationAvailability();
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
```

```javascript
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
  syncCalibrationAvailability();
}
```

In `setAnalyzingUiState()` and `setDoneUiState()`, retain their existing status,
Start, and Stop assignments, then call `syncCalibrationAvailability()` as the
final statement. Immediately after setting `state.phase = "recording"` in both
recording functions, call `syncCalibrationAvailability()`.

At the beginning of the **Save Sample** click handler, fail closed for synthetic
or stale clicks:

```javascript
if (!hasCatalogAuthority() || document.getElementById("save-sample").disabled) return;
```

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
pytest -q tests/test_phrase_api.py tests/test_websocket_flow.py -k "phrase or frontend"
npm run test:e2e
! rg -n "中文" frontend
```

Expected: all focused backend tests and all Playwright camera tests pass, and the
frontend scan finds no fixed `中文` label. The
initial-load test records zero session, WebSocket, and camera side effects; the
race test retains `zh_light_on_hello` after the deferred HTTP response finishes.

- [ ] **Step 9: Run full regression verification**

Run:

```bash
ruff check .
pytest -q
npm run test:e2e
git diff --check
```

Expected: Ruff is clean, the complete pytest suite passes with only existing
environment-gated skips, all Playwright tests pass, and Git reports no whitespace
errors.

- [ ] **Step 10: Commit the frontend task**

```bash
git add frontend/index.html frontend/websocket.js tests/test_websocket_flow.py tests/e2e/camera.spec.js
git commit -m "feat: preload phrase catalog in English UI"
```

---

## Manual Acceptance Check

After the two task commits are deployed to the Radeon host:

- [ ] Open the demo page in a fresh browser tab and confirm both language controls display `Chinese` and `English`.
- [ ] Confirm the default Phrase list shows the two Chinese catalog phrases plus `UNKNOWN` before any button click.
- [ ] Confirm the browser has not requested camera permission before **Start** or **Save Sample**.
- [ ] Switch Recording language to `English` and confirm only the two English phrases plus `UNKNOWN` are shown.
- [ ] Select a phrase, click **Save Sample**, and confirm camera/countdown begins only after the selection.
- [ ] Run one recognition clip and confirm the existing `clip.start` language and result flow still work.
