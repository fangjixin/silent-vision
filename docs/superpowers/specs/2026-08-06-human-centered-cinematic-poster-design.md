# Human-Centered Cinematic Poster Design

**Date:** 2026-08-06

## Purpose

Replace the current report-like poster with a human-centered promotional poster for the Track 1 submission. The poster must communicate, at a glance, who Silent Vision helps and what the product does. It supplements rather than replaces the separate six-page Project Profile PDF.

## Submission Roles

- `submission/Silent-Vision-Project-Profile.pdf` remains the required project profile. It explains the background, target users, scenarios, architecture, model, algorithm, and AMD Radeon/ROCm adaptation.
- `submission/Silent-Vision-Poster.pdf` is the printable A3 product poster.
- `submission/Silent-Vision-Poster.png` is the raster preview of the same poster for repository and Pull Request viewing.

## Audience and Product Positioning

Silent Vision is designed for people who can form visible mouth movements but cannot or prefer not to speak aloud, including:

- people with aphasia, voice disorders, or temporary post-surgical loss of speech;
- patients communicating fixed needs in hospitals or rehabilitation settings;
- people with hearing or speech disabilities using basic communication or device control;
- people operating devices in noisy, silence-required, or voice-inappropriate environments.

The poster must present these people with dignity and agency. It must not use pity, medical sensationalism, or exaggerated disability stereotypes.

## Creative Direction

Use a restrained, hopeful cinematic style. The composition is a portrait A3 `2 x 2` mosaic of four large photorealistic scenes. All people are fictional, and the ensemble is internationally diverse in ethnicity, gender, and age.

The four scenes are:

1. **POST-SURGERY** - A patient in a hospital room faces a tablet camera and silently forms a fixed request.
2. **REHABILITATION** - A person with a voice disorder works with a therapist and confirms communication through a screen.
3. **ACCESSIBLE COMMUNICATION** - A person with a hearing or speech disability communicates at a public-service counter through visible mouth movement and an assistive display.
4. **SILENT CONTROL INPUT** - A person in a noisy or silence-required environment forms a visible command at a camera terminal. The interface shows a classifier decision that a separately integrated system could map to lighting or equipment control; the scene must not show Silent Vision directly operating a device.

Every scene must keep the face, mouth, and environment legible. A subtle red mouth-recognition accent may connect the imagery to the product. Avoid science-fiction HUD clutter.

## Layout and Visual Hierarchy

- Four large photographs occupy roughly 70 percent of the poster.
- The central title crosses the intersection of the four photographs on a dark translucent band.
- A compact lower panel identifies the product boundary, AMD Radeon/ROCm path, and repository QR code.
- Use deep blue-black shadows, restrained skin tones, warm practical light, and AMD red as the accent color.
- Use generous typography and short copy. Do not reuse the report's card-grid visual language.

## Approved English Copy

Product name:

> SILENT VISION

Visible product qualifier:

> PERSONALIZED FIXED-PHRASE PROTOTYPE

Main title:

> A VOICE WITHOUT SOUND.

Audience statement:

> Visual communication for people who can form words but cannot speak them aloud.

Scene labels:

- `POST-SURGERY`
- `REHABILITATION`
- `ACCESSIBLE COMMUNICATION`
- `SILENT CONTROL INPUT`

Honest product boundary:

> Four registered phrases · Chinese + English · Exact phrase or safe UNKNOWN

Technical footer:

> Camera-only · No audio capture · ROCm PyTorch on AMD Radeon

The poster must not include the earlier example line `“Please turn on the light.” → LIGHT_ON`.

## Asset Production

Generate four coordinated photorealistic scenes without embedded text. Text, scene labels, title, footer, and QR code are added deterministically by the poster-generation code so that all English is exact and legible.

The implementation updates:

- `docs/submission/poster-copy.md`;
- `scripts/generate_submission_assets.py`;
- `tests/test_submission_docs.py`;
- `submission/Silent-Vision-Poster.pdf`;
- `submission/Silent-Vision-Poster.png`;
- `output/pdf/Silent-Vision-Poster.pdf`.

Generated source imagery must be stored inside the repository so rebuilding the poster does not depend on network access or a model call.

## Accuracy and Safety Boundaries

- Do not claim open-vocabulary lipreading, transcription, cross-speaker generalization, or measured bilingual accuracy.
- The four photographed people represent target-user scenarios, not demonstrated cross-speaker performance. The visible `PERSONALIZED FIXED-PHRASE PROTOTYPE` qualifier must remain on the poster.
- Do not imply that the prototype replaces clinical communication systems, sign language, speech-generating devices, or professional care.
- Do not show Silent Vision completing a device action. The control scene may show only a classifier decision intended for a separately integrated control system.
- Do not show a microphone as part of the recognition path.
- Do not obscure the mouth with a mask, hand, visor, or medical equipment.
- Keep the wording consistent with the fixed-phrase bilingual classifier and fail-closed `UNKNOWN` behavior.

## Verification

Before delivery:

1. Inspect every generated person for facial, mouth, hand, and anatomy defects.
2. Confirm the medical and public-service scenes are plausible and respectful.
3. Confirm all four scenarios remain readable at poster-preview scale.
4. Verify every line of English against the approved copy.
5. Verify the QR code resolves to the source repository.
6. Render the final PDF to PNG and inspect for clipping, overlap, weak contrast, or raster artifacts.
7. Confirm the PDF is one A3 portrait page and the PNG matches it visually.
8. Replace the obsolete poster tests that require the removed light-command example and old three-card layout. The new contract must require the approved title, product qualifier, four scene labels, product boundary, technical footer, and absence of the removed example line.
9. Run the updated submission-asset and documentation tests, including checks that all repository-owned source images exist and the generated poster can be rebuilt without network access.

## Non-Goals

- Redesigning the Project Profile PDF.
- Adding model-performance claims before final evaluation.
- Changing product behavior, phrase catalogs, training, or inference.
- Producing a generic architecture infographic or another report-style page.
