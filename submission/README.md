# Silent Vision - Track 1 Submission Index

Applicant: Jixin Fang

Pull request title: `Track 1, Jixin Fang, Silent Vision`

Source repository: <https://github.com/fangjixin/silent-vision>

Silent Vision is a personalized fixed-phrase visual classifier with one
four-phrase bilingual catalog: Chinese `你好，请帮我打开灯` and `你吃饭了吗？`, plus
English `Hello, please turn on the light.` and `Have you eaten?`. The user
selects the language before recording; the application does not infer language
from the clip. Its ROCm-only Torch classifier uses a selected-language softmax,
then returns exact catalog text and a mapped intent only when the probability and
phrase-centroid distance gates pass. Otherwise it returns heuristic `UNKNOWN`.
It is not open-vocabulary lipreading.

| Requirement | Repository path | Status |
| --- | --- | --- |
| Environment, dependencies, and startup guide | [`../README.md`](../README.md) | Complete |
| AMD Radeon / ROCm runbook | [`../docs/runbooks/amd-real-mode.md`](../docs/runbooks/amd-real-mode.md) | Complete |
| Phrase catalog | [`../command/phrase_catalog.json`](../command/phrase_catalog.json) | Complete |
| Project profile PDF | [`Silent-Vision-Project-Profile.pdf`](Silent-Vision-Project-Profile.pdf) | Complete |
| Project profile source | [`../docs/submission/project-profile-source.md`](../docs/submission/project-profile-source.md) | Complete |
| Poster PDF | [`Silent-Vision-Poster.pdf`](Silent-Vision-Poster.pdf) | Complete |
| Poster PNG | [`Silent-Vision-Poster.png`](Silent-Vision-Poster.png) | Complete |
| Poster source copy | [`../docs/submission/poster-copy.md`](../docs/submission/poster-copy.md) | Complete |
| Demo script and shot checklist | [`demo-video-script.md`](demo-video-script.md) | Ready for recording |
| Demo video | External URL to be added after recording | Pending |
| Pull request description | [`pull-request-description.md`](pull-request-description.md) | Ready; demo link pending |

The source, catalog, documentation, and generated assets are present. English
recordings, bilingual training, the official Radeon run, final evaluation report,
and video are still pending. No bilingual accuracy or other performance number
is claimed until the final report exists. If only a small-data smoke is
available, it must be described as proof of execution rather than evaluation.

Recordings, checkpoints, and private reports are intentionally excluded from the
public contest bundle.
