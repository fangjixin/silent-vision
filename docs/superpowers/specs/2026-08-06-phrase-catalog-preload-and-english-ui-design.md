# Phrase Catalog Preload and English UI Design

**Date:** 2026-08-06  
**Application:** Silent Vision  
**Status:** Approved for implementation planning

## 1. Objective

Make the recording controls usable immediately after the page loads and keep all
fixed interface text in English. The Phrase selector must show the active
language's registered phrases before the user starts a recording, without opening
a WebSocket or requesting camera access.

Chinese phrases remain in Chinese because they are recognition and training data,
not interface labels.

## 2. Current Problem

The browser currently receives the phrase catalog only in the WebSocket
`session.ready` event. The frontend does not create a session or WebSocket until
the user clicks **Start** or **Save Sample**, so the Phrase selector is empty and
disabled on initial page load. This prevents the user from reviewing and choosing
the phrase before recording begins.

The language selectors also display the interface label `中文`, which conflicts
with the contest requirement that submission materials and the demonstration
interface be in English.

## 3. Chosen Approach

Add a read-only HTTP endpoint and load the catalog when the page starts.

`GET /api/phrases` returns the records from `app.state.phrase_catalog`. This is the
same active catalog used by the running classifier:

- Prototype mode uses the repository phrase catalog.
- Torch mode uses the catalog embedded in the loaded checkpoint.

The endpoint therefore does not maintain a second catalog or infer labels from
frontend data.

Opening the page starts one catalog request only. It does not create a recognition
session, open a WebSocket, request camera permission, or begin a countdown. Those
actions remain tied to **Start** and **Save Sample**.

### Alternatives rejected

Opening a WebSocket on page load would avoid a new endpoint, but would create idle
sessions and require reconnect behavior before the user has chosen an action.

Embedding the catalog in the HTML would load it without another request, but would
couple runtime model data to a static page and could allow the displayed phrases
to drift from a Torch checkpoint's catalog.

## 4. API Contract

`GET /api/phrases` returns HTTP 200 with this shape:

```json
{
  "phrases": [
    {
      "phraseId": "zh_light_on_hello",
      "text": "你好，请帮我打开灯",
      "language": "zh",
      "intent": "LIGHT_ON",
      "enabled": true
    }
  ]
}
```

The records use the existing `catalog_records` serialization. The endpoint is
read-only and has no session lifecycle. The existing `session.ready` catalog is
retained because it describes the catalog used by that WebSocket runtime; this
change does not alter the recording or inference protocol.

## 5. Frontend Behavior and Data Flow

On module initialization, the frontend:

1. Disables **Save Sample** and displays a disabled `Loading phrases...` option.
2. Requests `GET /api/phrases`.
3. Stores the returned records in the existing frontend catalog state.
4. Filters enabled records by the selected recording language.
5. Renders those phrase texts followed by the existing `UNKNOWN` choice.
6. Enables the Phrase selector and **Save Sample**.

The default recording language is `zh`, displayed as `Chinese`, so a successful
initial response shows the two registered Chinese phrases. Changing the recording
language to `en`, displayed as `English`, immediately re-renders only the English
phrases plus `UNKNOWN`.

The catalog state records both the records and their authority: `loading`,
`http`, `websocket`, or `error`. A valid HTTP response changes `loading` to
`http`. A `session.ready` event always replaces the records, changes the authority
to `websocket`, and re-renders the options before recording. Once the authority is
`websocket`, a late HTTP success or failure is ignored for the rest of the page
lifetime. This prevents a request started at page load from overwriting the
catalog actually used by the runtime session.

When either valid source re-renders the options, the current phrase remains
selected if its ID still exists. Otherwise the first enabled phrase for the
selected language becomes selected; if that language has no registered phrase,
`UNKNOWN` becomes selected.

Clicking **Save Sample** after loading follows the existing flow: create a session,
connect the WebSocket, wait for `session.ready`, request camera access, count down,
record, and upload the selected sample.

## 6. Loading and Error States

While the HTTP request is pending, the Phrase selector remains disabled and shows
`Loading phrases...`; **Save Sample** remains disabled.

If the request fails or has an invalid payload before a `session.ready` event has
provided a valid catalog, the selector remains disabled and shows
`Unable to load phrases`. **Save Sample** remains disabled so an empty or stale
phrase identifier cannot be recorded. A later valid `session.ready` event may
recover this state. A late HTTP failure cannot replace an existing `websocket`
state. The recognition **Start** control remains independent of this
calibration-only loading state.

A valid catalog with no registered phrase for the selected language is not a
loading error. The selector contains only `UNKNOWN`, enables its custom phrase
input, and permits **Save Sample**. This supports language-specific unrelated
samples used to calibrate rejection.

All code paths that change the stream phase or catalog state call one shared
calibration-availability function. That function enables **Save Sample** only
when a valid catalog source exists and no recording or analysis is active.
Stopped, cancelled, done, loading, error, and WebSocket-close transitions must not
enable the control unconditionally.

A page reload retries the request. This design does not add background retries or
a separate retry button.

## 7. English Interface Boundary

Both language selectors use these visible labels while retaining their existing
protocol values:

| Visible label | Value |
| --- | --- |
| `Chinese` | `zh` |
| `English` | `en` |

All fixed labels, buttons, loading text, status text, validation messages, and
error text in the demonstration interface must be English. Registered phrase
content remains in its original language. Therefore Chinese catalog phrases may
appear in the Phrase selector, result payloads, and sample metadata without
violating the English-interface requirement.

## 8. Testing

Backend tests verify that `GET /api/phrases` returns the active catalog and uses
the existing record schema.

Frontend and end-to-end tests verify that:

- both language selectors display `Chinese` and `English`;
- the initial page load requests and renders the Chinese phrase list;
- loading the catalog does not create a session, open a WebSocket, or request the
  camera;
- changing the recording language renders the matching English or Chinese
  phrases plus `UNKNOWN`;
- **Save Sample** is disabled during loading and enabled after a valid response;
- request failure or an invalid/empty global response produces the English
  failure state and prevents calibration recording;
- a language with no registered phrase renders only `UNKNOWN` and permits an
  unrelated-phrase recording;
- `session.ready` supersedes a pending HTTP request, and a late HTTP success or
  failure cannot replace its catalog or disable its valid calibration state;
- stopped, cancelled, done, and WebSocket-close transitions cannot bypass the
  catalog-state gate for **Save Sample**; and
- the existing recording payload still sends the selected `language` and
  `phraseId`.

## 9. Acceptance Criteria

- After a normal page refresh, the Phrase selector becomes usable without any
  button click.
- The user can choose a language and phrase before camera permission or recording
  begins.
- Catalog loading alone creates no session and no WebSocket connection.
- The WebSocket catalog remains authoritative when HTTP and WebSocket responses
  arrive out of order.
- A selected language with no registered phrase can record `UNKNOWN` but cannot
  record a known phrase.
- Every fixed interface string is English.
- Chinese recognition phrases remain unchanged.
- Prototype and Torch modes both display the exact catalog active in that running
  backend.
