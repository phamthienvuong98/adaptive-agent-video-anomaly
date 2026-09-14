"""
main.py
────────
Entry point của hệ thống VAD Agent.
Chạy demo, evaluation, hoặc real-time inference.
"""

import argparse
import sys
from pathlib import Path

import cv2
from loguru import logger

from agents.orchestrator import VADOrchestrator
from utils.alert_engine import Alert


def parse_args():
    p = argparse.ArgumentParser(description="VAD Agent — Phát hiện hành vi bất thường")
    p.add_argument("--source", default="0", help="Video file hoặc camera index (default: 0)")
    p.add_argument("--config", default="configs/agent_config.yaml", help="Config file")
    p.add_argument("--mode", choices=["demo", "eval", "calibrate", "api"], default="demo")
    p.add_argument("--camera-id", default="cam_01")
    p.add_argument("--normal-video", help="Video bình thường cho calibration")
    p.add_argument("--output", default="outputs/", help="Thư mục output")
    p.add_argument("--show", action="store_true", help="Hiển thị video realtime")
    return p.parse_args()


def run_demo(args):
    """Demo mode: xử lý video và hiển thị alert."""
    orchestrator = VADOrchestrator(config_path=args.config)
    source = int(args.source) if args.source.isdigit() else args.source

    logger.info(f"Demo mode | source={source} | camera={args.camera_id}")

    alert_count = 0
    for alert in orchestrator.process_video(source):
        if alert:
            alert_count += 1
            print(f"\n{'='*60}")
            print(alert)
            print(f"{'='*60}\n")

    stats = orchestrator.get_stats()
    print(f"\n📊 Kết quả: {stats.summary()}")
    print(f"   Alerts: {alert_count}")


def run_calibrate(args):
    """Calibrate adaptive threshold từ video bình thường."""
    if not args.normal_video:
        logger.error("Cần cung cấp --normal-video cho calibration mode")
        sys.exit(1)

    orchestrator = VADOrchestrator(config_path=args.config)
    orchestrator.calibrate(
        normal_video_path=args.normal_video,
        seconds=120,
    )
    logger.info("Calibration hoàn thành. Hệ thống đã sẵn sàng.")


def run_api(args):
    """Khởi động FastAPI server để nhận video stream qua HTTP."""
    try:
        import uvicorn
        from fastapi import FastAPI, UploadFile
        import numpy as np

        app = FastAPI(title="VAD Agent API")
        orchestrator = VADOrchestrator(config_path=args.config)

        @app.post("/analyze-frame")
        async def analyze_frame(file: UploadFile):
            import io
            data = await file.read()
            arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            from utils.video_preprocessor import FrameData
            import time
            fd = FrameData(frame=frame, index=0, timestamp=time.time(), motion_score=0.0)
            alert = orchestrator.process_frame(fd)
            if alert:
                return alert.to_dict()
            return {"is_anomaly": False}

        @app.get("/stats")
        def get_stats():
            s = orchestrator.get_stats()
            return {"fps": s.fps, "pass_rate": s.pass_rate, "alerts": s.alerts_fired}

        uvicorn.run(app, host="0.0.0.0", port=8000)

    except ImportError:
        logger.error("Cài fastapi và uvicorn: pip install fastapi uvicorn")
        sys.exit(1)


def main():
    args = parse_args()
    Path(args.output).mkdir(exist_ok=True)

    if args.mode == "demo":
        run_demo(args)
    elif args.mode == "calibrate":
        run_calibrate(args)
    elif args.mode == "api":
        run_api(args)
    elif args.mode == "eval":
        logger.info("Eval mode: xem tests/ để chạy benchmark")


if __name__ == "__main__":
    main()
