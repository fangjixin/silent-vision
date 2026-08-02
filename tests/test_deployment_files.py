from pathlib import Path


def test_docker_compose_mounts_persistence_root_and_rocm_devices():
    compose = Path("docker/docker-compose.yml").read_text()
    assert "/workspace/persistent/silent-vision:/workspace/persistent/silent-vision" in compose
    assert "/dev/kfd:/dev/kfd" in compose
    assert "/dev/dri:/dev/dri" in compose
    assert "127.0.0.1:8000:8000" in compose


def test_dockerfile_does_not_copy_model_weights():
    dockerfile = Path("docker/Dockerfile").read_text()
    assert "COPY . /app" in dockerfile
    assert "models/" not in dockerfile


def test_smoke_scripts_exist_and_are_executable():
    fake = Path("scripts/smoke_fake.sh")
    rocm = Path("scripts/smoke_rocm.sh")
    setup = Path("scripts/setup_amd_real.sh")
    start = Path("scripts/start_real_rocm.sh")
    oneclick = Path("scripts/amd_real_oneclick.sh")
    assert fake.exists()
    assert rocm.exists()
    assert setup.exists()
    assert start.exists()
    assert oneclick.exists()
    assert fake.read_text().startswith("#!/usr/bin/env bash")
    assert rocm.read_text().startswith("#!/usr/bin/env bash")
    assert setup.read_text().startswith("#!/usr/bin/env bash")
    assert start.read_text().startswith("#!/usr/bin/env bash")
    assert oneclick.read_text().startswith("#!/usr/bin/env bash")
    assert "Visual_Speech_Recognition_for_Multiple_Languages" in rocm.read_text()
    assert "models/avhubert/model.pt" not in rocm.read_text()
    assert "/opt/venv/bin/python" in setup.read_text()
    assert "/opt/venv/bin/python" in start.read_text()
    assert "/workspace/persistent/silent-vision" in setup.read_text()
    assert "/workspace/persistent/silent-vision" in start.read_text()
    assert "scripts/setup_amd_real.sh" in oneclick.read_text()
    assert "scripts/start_real_rocm.sh" in oneclick.read_text()
    assert "rc-tunnel" not in oneclick.read_text()
    assert "huggingface-hub>=0.30.0,<1.0.0" in setup.read_text()
    assert "pip install --upgrade huggingface_hub" not in setup.read_text()
    assert "--force-reinstall --no-deps" in setup.read_text()
    assert "check_transformers_stack" in setup.read_text()
    assert "MPC001_LM_WEIGHT" in setup.read_text()
    assert "decode.lm_weight" in setup.read_text()
    assert "download_if_missing_or_invalid_zip" in setup.read_text()
    assert "valid zip exists; skip download" in setup.read_text()
    assert "huggingface-hub must be <1.0" in start.read_text()
    assert "MPC001_ENGLISH_CONFIG" in start.read_text()
    assert "MPC001_CHINESE_CONFIG" in start.read_text()


def test_frontend_stream_lifecycle_cleans_up_between_starts():
    websocket_js = Path("frontend/websocket.js").read_text()
    index_html = Path("frontend/index.html").read_text()
    assert '<button id="stopButton" type="button" disabled>Cancel</button>' in index_html
    assert "cleanupCurrentStream();" in websocket_js
    assert "setStoppedUiState();" in websocket_js
    assert "setText(\"visionStatus\", \"stopped\");" in websocket_js
    assert "setText(\"lipStatus\", \"stopped\");" in websocket_js
    assert "setText(\"semanticStatus\", \"stopped\");" in websocket_js
    assert "setText(\"agentStatus\", \"stopped\");" in websocket_js
    assert "setText(\"candidateOutput\", \"\");" in websocket_js
    assert "setText(\"resultOutput\", \"\");" in websocket_js
    assert "connectionGeneration" in websocket_js
    assert "socketGeneration !== state.connectionGeneration" in websocket_js
    assert "state.ws.close(1000, \"client stopped stream\");" in websocket_js
    assert "state.ws.bufferedAmount > 2_000_000" in websocket_js
    assert "state.streaming = false;" in websocket_js
    assert "mouth reused" in websocket_js


def test_frontend_one_shot_capture_auto_submits_full_buffer_without_cancelling_inference():
    websocket_js = Path("frontend/websocket.js").read_text()
    camera_js = Path("frontend/camera.js").read_text()
    assert "phase: \"idle\"" in websocket_js
    assert "autoSubmitWhenBufferFull(event);" in websocket_js
    assert "function autoSubmitWhenBufferFull(event)" in websocket_js
    assert "runCaptureCountdown(connectionGeneration);" in websocket_js
    assert "captureCountdownSeconds" in websocket_js
    assert "await state.camera.startPreview();" in websocket_js
    assert "state.camera.startCapture();" in websocket_js
    assert "async startPreview()" in camera_js
    assert "startCapture()" in camera_js
    assert "setAnalyzingUiState();" in websocket_js
    assert "setDoneUiState();" in websocket_js
    assert "cameraStatus\", \"recorded\"" in websocket_js
    assert "semanticStatus\", \"running\"" in websocket_js
    assert "agentStatus\", \"done\"" in websocket_js
    assert "if (state.phase !== \"recording\") return;" in websocket_js
    assert "event.bufferedFrames < event.requiredFrames" in websocket_js
    assert "stopCamera();" in websocket_js
    assert "state.streaming = false;" in websocket_js
    assert "state.ws.send(JSON.stringify({ type: \"stream.stop\" }));" in websocket_js
    assert "state.ws.send(JSON.stringify({ type: \"stream.commit\" }));" in websocket_js
    assert "function autoSubmitWhenBufferFull" in websocket_js
    auto_submit_block = websocket_js.split("function autoSubmitWhenBufferFull", 1)[1].split("function ", 1)[0]
    assert "stream.stop" not in auto_submit_block
    assert "stream.commit" in auto_submit_block


def test_frontend_records_utterance_level_binary_video_clips():
    websocket_js = Path("frontend/websocket.js").read_text()
    camera_js = Path("frontend/camera.js").read_text()
    assert "MediaRecorder" in camera_js
    assert "recordClip" in camera_js
    assert "video/webm" in camera_js
    assert "clip.start" in websocket_js
    assert "state.ws.send(clipBlob);" in websocket_js
    assert "readAsDataURL" not in camera_js
    assert "base64" not in camera_js.lower()


def test_backend_uses_pyav_25fps_clip_preprocessing_and_debug_artifacts():
    requirements = Path("requirements.txt").read_text()
    websocket_py = Path("api/websocket.py").read_text()
    clip_py = Path("video/clip.py").read_text()
    roi_py = Path("video/mouth_roi.py").read_text()
    assert "av>=" in requirements
    assert "decode_video_clip" in websocket_py
    assert "target_fps" in clip_py
    assert "command_clip_fps" in websocket_py
    assert "write_mouth_roi_video" in roi_py
    assert "write_aligned_face_video" in roi_py
    assert "mouth_roi_video" in websocket_py
    assert "mouth_roi_npy" in websocket_py
    assert "aligned_face_video" in websocket_py
    assert "original_video" in websocket_py


def test_command_classifier_files_and_scripts_exist():
    for path in [
        "command/labels.py",
        "command/model.py",
        "command/inference.py",
        "scripts/train_command_classifier.py",
        "scripts/validate_command_classifier.py",
        "scripts/infer_command_clip.py",
        "scripts/record_command_manifest.py",
    ]:
        assert Path(path).exists()

    model_py = Path("command/model.py").read_text()
    inference_py = Path("command/inference.py").read_text()
    schemas_py = Path("backend/schemas.py").read_text()
    assert "class CommandConformerClassifier" in model_py
    assert "num_layers: int = 4" in model_py
    assert "AttentivePooling" in model_py
    assert "depthwise_conv" in model_py
    assert "top1_margin" in inference_py
    assert "CommandDecision" in schemas_py
    assert "LIGHT_ON" in Path("command/labels.py").read_text()


def test_rejected_commands_do_not_call_llm_or_execute_actions():
    websocket_py = Path("api/websocket.py").read_text()
    agent_py = Path("agent/agent.py").read_text()
    assert "command_decision.accepted" in websocket_py
    command_clip_block = websocket_py.split("async def process_command_clip", 1)[1].split("async def run_inference_loop", 1)[0]
    assert "semantic_interpreter.interpret" not in command_clip_block
    assert "decide_command" in agent_py
    assert "action=\"reject\"" in agent_py


def test_websocket_has_calibration_upload_path():
    source = Path("api/websocket.py").read_text()

    assert "calibration.start" in source
    assert "calibration.saved" in source
    assert "save_prototype_sample" in source
    assert "profileId" in source


def test_backend_stream_stop_cancels_running_inference_and_ignores_stale_windows():
    websocket_py = Path("api/websocket.py").read_text()
    manager_py = Path("session/manager.py").read_text()
    inference_py = Path("lip/inference.py").read_text()
    mpc001_py = Path("lip/mpc001.py").read_text()
    assert "def stop_stream(self) -> None:" in manager_py
    assert "stream_generation" in manager_py
    assert "inference_cancel_event" in manager_py
    assert "active.stop_stream()" in websocket_py
    assert "active.commit_stream()" in websocket_py
    assert "active.accepting_frames" in websocket_py
    assert "stream.committed" in websocket_py
    assert "cancel_active_inference(" in websocket_py
    assert "cancel_active_inference(\"websocket cleanup\")" in websocket_py
    assert "active.inference_cancel_event.set()" in websocket_py
    assert "stream_generation" in websocket_py
    assert "is_live_stream" in websocket_py
    assert "cancel_event" in inference_py
    assert "subprocess.Popen" in mpc001_py
    assert "start_new_session=True" in mpc001_py
    assert "os.killpg" in mpc001_py
    assert "subprocess.run" not in mpc001_py
    assert "mpc001 subprocess cancelled" in mpc001_py


def test_backend_debug_dump_writes_raw_frame_overlay_contact_sheet():
    websocket_py = Path("api/websocket.py").read_text()
    base_py = Path("lip/base.py").read_text()
    assert "debug_image" in base_py
    assert "debug_mouth_box" in base_py
    assert "debug_landmarks" in base_py
    assert "captureCountdownSeconds" in websocket_py
    assert "_dump_raw_debug_window" in websocket_py
    assert "raw_png" in websocket_py
    assert "ImageDraw.Draw" in websocket_py
    assert "debug_image=image.copy()" in websocket_py


def test_real_mode_suppresses_noisy_third_party_warnings():
    minicpm = Path("llm/minicpm.py").read_text()
    face = Path("vision/face.py").read_text()
    assert "image_processor_class argument is deprecated" in minicpm
    assert "Using a slow image processor" in minicpm
    assert "SymbolDatabase.GetPrototype" in face


def test_requirements_use_current_compatible_real_mode_versions():
    requirements = Path("requirements.txt").read_text()
    assert "fastapi>=0.141.1,<1.0.0" in requirements
    assert "uvicorn[standard]>=0.52.0,<1.0.0" in requirements
    assert "pydantic>=2.13.4,<3.0.0" in requirements
    assert "transformers==4.51.0" in requirements
    assert "huggingface-hub>=0.30.0,<1.0.0" in requirements
    assert "numpy>=1.26.4,<2.0.0" in requirements
    assert "opencv-python-headless>=4.10.0,<4.11.0" in requirements
    assert "mediapipe==0.10.14" in requirements
    assert "minicpmo-utils==1.0.6" in requirements
    assert "Pillow==10.4.0" in requirements
    assert "Pillow>=12.0.0" not in requirements
    assert "librosa==0.9.0" in requirements
    assert "soundfile==0.12.1" in requirements
    assert "torch" not in requirements
