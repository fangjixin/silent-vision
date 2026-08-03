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
    assert "Visual_Speech_Recognition_for_Multiple_Languages" not in rocm.read_text()
    assert "models/avhubert/model.pt" not in rocm.read_text()
    assert "/opt/venv/bin/python" in setup.read_text()
    assert "/opt/venv/bin/python" in start.read_text()
    assert "/workspace/persistent/silent-vision" in setup.read_text()
    assert "/workspace/persistent/silent-vision" in start.read_text()
    assert "scripts/setup_amd_real.sh" in oneclick.read_text()
    assert "scripts/start_real_rocm.sh" in oneclick.read_text()
    assert "rc-tunnel" not in oneclick.read_text()
    for removed in ["Visual_Speech_Recognition_for_Multiple_Languages", "models/minicpm-o-4_5", "HF_HOME", "MPC001_REPO"]:
        assert removed not in setup.read_text()
        assert removed not in start.read_text()
    assert "check_no_removed_dependencies" in setup.read_text()
    assert "COMMAND_CLASSIFIER_CHECKPOINT" in start.read_text()


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
    assert "state.streaming = false;" in websocket_js


def test_frontend_one_shot_capture_sends_single_binary_clip():
    websocket_js = Path("frontend/websocket.js").read_text()
    camera_js = Path("frontend/camera.js").read_text()
    assert "phase: \"idle\"" in websocket_js
    assert "runCaptureCountdown(connectionGeneration);" in websocket_js
    assert "captureCountdownSeconds" in websocket_js
    assert "await state.camera.startPreview();" in websocket_js
    assert "async startPreview()" in camera_js
    assert "MediaRecorder" in camera_js
    assert "setAnalyzingUiState();" in websocket_js
    assert "setDoneUiState();" in websocket_js
    assert "cameraStatus\", \"recorded\"" in websocket_js
    assert "agentStatus\", \"done\"" in websocket_js
    assert "stopCamera();" in websocket_js
    assert "state.streaming = false;" in websocket_js
    assert "state.ws.send(clipBlob);" in websocket_js
    assert "state.ws.send(JSON.stringify({ type: \"stream.stop\" }));" in websocket_js
    assert "function autoSubmitWhenBufferFull" not in websocket_js
    assert "windowFrames" not in websocket_js


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
    assert "image/jpeg" not in camera_js


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
    assert "semantic_interpreter" not in websocket_py
    assert "MiniCPM" not in websocket_py
    assert "decide_command" in agent_py
    assert "action=\"reject\"" in agent_py


def test_websocket_has_calibration_upload_path():
    source = Path("api/websocket.py").read_text()

    assert "calibration.start" in source
    assert "calibration.saved" in source
    assert "save_prototype_sample" in source
    assert "profileId" in source


def test_frontend_has_prototype_calibration_ui():
    html = Path("frontend/index.html").read_text()
    js = Path("frontend/websocket.js").read_text()

    assert "calibration-intent" in html
    assert "calibration-phrase" in html
    assert "Save Sample" in html
    assert "GLOBAL_PROFILE_ID = \"global\"" in js
    assert "silentVisionProfileId" not in js
    assert "localStorage" not in js
    assert "profileId: GLOBAL_PROFILE_ID" in js
    assert "scope: \"global\"" in js
    assert "calibration.start" in js


def test_prototype_scripts_and_startup_defaults_exist():
    assert Path("scripts/inspect_prototypes.py").exists()
    assert Path("scripts/build_global_prototypes.py").exists()

    oneclick = Path("scripts/amd_real_oneclick.sh").read_text()
    readme = Path("README.md").read_text()

    assert "COMMAND_BACKEND=prototype" in oneclick
    assert "profiles/global" in readme
    assert "profileId=global" in readme
    assert "Personal Profile" not in readme


def test_backend_clip_cancel_and_cleanup_release_active_session():
    websocket_py = Path("api/websocket.py").read_text()
    manager_py = Path("session/manager.py").read_text()
    assert "def stop_stream(self) -> None:" in manager_py
    assert "stream_generation" in manager_py
    assert "inference_cancel_event" in manager_py
    assert "active.stop_stream()" in websocket_py
    assert "active.commit_stream()" in websocket_py
    assert "active.accepting_frames" in websocket_py
    assert "cancel_active_inference(" in websocket_py
    assert "cancel_active_inference(\"websocket cleanup\")" in websocket_py
    assert "active.inference_cancel_event.set()" in websocket_py
    assert "clip.cancel" in websocket_py
    assert "run_inference_loop" not in websocket_py
    assert "add_mouth_frame" not in manager_py


def test_backend_debug_dump_writes_clip_artifacts():
    websocket_py = Path("api/websocket.py").read_text()
    assert "captureCountdownSeconds" in websocket_py
    assert "save_original_video" in websocket_py
    assert "mouth_roi_npy" in websocket_py
    assert "aligned_face_video" in websocket_py
    assert "mouth_roi_video" in websocket_py


def test_real_mode_suppresses_noisy_mediapipe_warnings():
    face = Path("vision/face.py").read_text()
    assert "SymbolDatabase.GetPrototype" in face


def test_requirements_use_current_compatible_real_mode_versions():
    requirements = Path("requirements.txt").read_text()
    assert "fastapi>=0.141.1,<1.0.0" in requirements
    assert "uvicorn[standard]>=0.52.0,<1.0.0" in requirements
    assert "pydantic>=2.13.4,<3.0.0" in requirements
    assert "numpy>=1.26.4,<2.0.0" in requirements
    assert "opencv-python-headless>=4.10.0,<4.11.0" in requirements
    assert "mediapipe==0.10.14" in requirements
    assert "Pillow==10.4.0" in requirements
    assert "Pillow>=12.0.0" not in requirements
    for removed in ["transformers", "huggingface-hub", "accelerate", "safetensors", "librosa", "soundfile", "minicpmo-utils"]:
        assert removed not in requirements
    assert "torch" not in requirements
