import os
import json
import cv2
import sounddevice as sd
import numpy as np
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple
from flask import Flask, request, render_template_string, jsonify, Response
import socket
import sys

# =================== SETTINGS ===================
# Primary model preference list (will try in order)
PRIMARY_MODEL = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "microsoft/phi-1_5")
MODEL_CANDIDATES = [PRIMARY_MODEL, FALLBACK_MODEL]

# Quantization / device settings
LOAD_IN_4BIT = False              # Try 4-bit quantization if possible (GPU + bitsandbytes)
FORCE_CPU = True                # Force CPU mode (even if GPU is available)
MAX_NEW_TOKENS = 32             # Keep generation short for speed (CPU safe)
TEMPERATURE = 0.1
TOP_P = 0.92
REPETITION_PENALTY = 1.05

# Vision settings
CAMERA_INDEX = 0
SHOW_WINDOW = False              # Set True if you want native OpenCV window; leave False for Flask stream
FACE_DETECT_EVERY = 5

# Audio settings
SAMPLERATE = 16000
CHUNK = 2048
RMS_THRESHOLD = 500.0            # Speaking threshold if no VAD
USE_VAD = True                   # Try to use WebRTC VAD if available

# Database path
DB_PATH = "memory.db"

# =================== OPTIONAL IMPORTS ===================
TRANSFORMERS_OK = False
BITSANDBYTES_OK = False
CUDA_OK = False
try:
    import torch
    CUDA_OK = torch.cuda.is_available()
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    TRANSFORMERS_OK = True
    try:
        import bitsandbytes as bnb  # noqa: F401
        BITSANDBYTES_OK = True
    except Exception:
        BITSANDBYTES_OK = False
except Exception:
    TRANSFORMERS_OK = False
    BITSANDBYTES_OK = False
    CUDA_OK = False

VAD_OK = False
try:
    import webrtcvad
    VAD_OK = True
except Exception:
    VAD_OK = False

# =================== BASE ===================
class BaseModule:
    def __init__(self, name: str):
        self.name = name

    def log(self, msg: str):
        print(f"[{self.name}] {msg}", flush=True)

    def start(self): ...
    def run_once(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]: ...
    def stop(self): ...

# =================== VISION ===================
class VisionModule(BaseModule):
    def __init__(self, camera_index: int = CAMERA_INDEX, show_window: bool = SHOW_WINDOW, face_detect_every: int = FACE_DETECT_EVERY):
        super().__init__("VisionModule")
        self.camera_index = camera_index
        self.show_window = show_window
        self.face_detect_every = face_detect_every
        self.cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._person_detected = False
        self._faces_to_draw = []
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            self.log(f"Cannot open camera index {self.camera_index}. Vision disabled.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.log("Vision thread started")

    def _capture_loop(self):
        frame_count = 0
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame_count += 1

            faces_draw = []
            person_detected = False

            if frame_count % self.face_detect_every == 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                    person_detected = len(faces) > 0
                    faces_draw = list(faces)
                except Exception:
                    person_detected = False
                    faces_draw = []

                with self._lock:
                    self._person_detected = person_detected
                    self._faces_to_draw = faces_draw

            if self.show_window:
                to_draw = []
                with self._lock:
                    to_draw = list(self._faces_to_draw)
                for (x, y, w, h) in to_draw:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow("AI Companion Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

            with self._lock:
                self._latest_frame = frame

        if self.cap:
            self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def run_once(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            frame = self._latest_frame
            person = self._person_detected
        if frame is None:
            return {"vision": {"ok": False, "reason": "no_frame"}}
        h, w = frame.shape[:2]
        return {"vision": {"ok": True, "data": f"frame {w}x{h}", "person_detected": bool(person)}}

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        try:
            if self.cap:
                self.cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
        except Exception:
            pass
        self.log("Vision stopped")

# =================== AUDIO ===================
class AudioModule(BaseModule):
    def __init__(self, samplerate=SAMPLERATE, chunk=CHUNK, speak_rms_threshold=RMS_THRESHOLD, use_vad=USE_VAD):
        super().__init__("AudioModule")
        self.samplerate = samplerate
        self.chunk = chunk
        self.speak_rms_threshold = speak_rms_threshold
        self._stream = None
        self._lock = threading.Lock()
        self._buffer = None
        self.use_vad = use_vad and VAD_OK
        self.vad = webrtcvad.Vad(2) if self.use_vad else None   # 0-3 (3 is aggressive)

    def start(self):
        try:
            self._stream = sd.InputStream(
                channels=1, samplerate=self.samplerate, blocksize=self.chunk,
                dtype="int16", callback=self._callback
            )
            self._stream.start()
            self.log(f"Audio stream started (VAD={'ON' if self.use_vad else 'OFF'})")
        except Exception as e:
            self.log(f"Audio start failed: {e}")

    def _callback(self, indata, frames, time_info, status):
        if status:  # audio overflows/underflows warnings
            pass
        with self._lock:
            self._buffer = indata.copy()

    def _vad_is_speech(self, samples: np.ndarray) -> bool:
        # WebRTC VAD expects 10/20/30 ms frames; use 30ms windows
        # 30ms @16kHz = 480 samples
        if len(samples) < 480:
            return False
        try:
            pcm16 = samples.astype(np.int16).tobytes()
            # We can check multiple 30ms chunks in the buffer for stability
            chunks = [pcm16[i:i+960] for i in range(0, min(len(pcm16), 960*3), 960)]  # up to ~90ms
            for ch in chunks:
                if len(ch) == 960 and self.vad.is_speech(ch, sample_rate=self.samplerate):
                    return True
            return False
        except Exception:
            return False

    def run_once(self, context):
        with self._lock:
            buf = self._buffer
            self._buffer = None
        if buf is None:
            return None

        buf_int16 = buf.astype(np.int16).squeeze()
        speaking = False
        rms = float(np.sqrt(np.mean(buf_int16.astype(np.float32) ** 2)))

        if self.use_vad:
            speaking = self._vad_is_speech(buf_int16)
        else:
            speaking = rms > self.speak_rms_threshold

        return {"audio": {"ok": True, "rms": rms, "speaking": bool(speaking), "vad": self.use_vad}}

    def stop(self):
        try:
            if self._stream:
                self._stream.stop(); self._stream.close()
        except Exception:
            pass
        self.log("Audio stopped")

# =================== MEMORY ===================
class MemoryModule(BaseModule):
    def __init__(self, db_path=DB_PATH):
        super().__init__("MemoryModule")
        self.db_path = db_path
        self.conn = None
        self._lock = threading.Lock()

    def start(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vision TEXT,
            audio TEXT
        )""")
        self.conn.commit()
        self.log("SQLite ready")

    def run_once(self, context):
        v = context.get("vision"); a = context.get("audio")
        if not v and not a:
            return None
        ts = datetime.now().isoformat()
        vs = json.dumps(v) if v else ""
        as_ = json.dumps(a) if a else ""
        with self._lock:
            self.conn.execute("INSERT INTO memory (timestamp, vision, audio) VALUES (?, ?, ?)",
                              (ts, vs, as_))
            self.conn.commit()

    def get_last_n(self, n=10) -> List[Tuple[str, str, str]]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT timestamp, vision, audio FROM memory ORDER BY id DESC LIMIT ?", (n,))
            return c.fetchall()

    def get_since(self, minutes=5):
        cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT timestamp, vision, audio FROM memory WHERE timestamp >= ?", (cutoff,))
            return c.fetchall()

    def stop(self):
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        self.log("SQLite closed")

# =================== CONTEXT ===================
class ContextModule(BaseModule):
    def __init__(self):
        super().__init__("ContextModule")
        self.last_summary = "No context yet"

    def run_once(self, context):
        v = context.get("vision") or {}; a = context.get("audio") or {}
        person = bool(v.get("person_detected", False))
        speaking = bool(a.get("speaking", False))
        ts = datetime.now().strftime("%H:%M:%S")
        if person and speaking:
            summary = f"At {ts}, a person was visible and speaking → conversation happening."
        elif person:
            summary = f"At {ts}, a person was visible but not speaking."
        elif speaking:
            summary = f"At {ts}, voice detected but no one visible."
        else:
            summary = f"At {ts}, no activity detected."
        self.last_summary = summary
        self.log(summary)
        return {"context_summary": summary}

# =================== REASONING ===================
class ReasoningModule(BaseModule):
    """
    Tries to load Phi-1.5 (or fallback TinyLlama) via transformers
    (with 4-bit quantization if possible).
    Falls back to a lightweight rule-based reasoning engine if models unavailable.
    """
    def __init__(self, memory: MemoryModule, context: ContextModule):
        super().__init__("ReasoningModule")
        self.memory = memory
        self.context = context
        self.tokenizer = None
        self.model = None
        self.model_id = None
        self.engine = "rule-based"
        self.loading = False
        self._lock = threading.Lock()

    def start(self):
        self.loading = True
        threading.Thread(target=self._load_best_model, daemon=True).start()

    def _load_best_model(self):
        if not TRANSFORMERS_OK:
            self.log("transformers not available → fallback to rule-based.")
            self.engine = "rule-based"
            self.loading = False
            return

        # Decide device + quantization
        use_gpu = CUDA_OK and (not FORCE_CPU)
        quant_cfg = None
        device_map = "cpu"
        torch_dtype = None

        if use_gpu and LOAD_IN_4BIT and BITSANDBYTES_OK:
            device_map = "auto"
            torch_dtype = torch.float16
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif use_gpu:
            device_map = "auto"
            torch_dtype = torch.float16  # fp16 on GPU
        else:
            device_map = "cpu"
            torch_dtype = None  # CPU full precision

        # Try list of models
        for candidate in MODEL_CANDIDATES:
            try:
                self.log(f"Attempting to load model: {candidate} with device_map={device_map}, 4bit={'yes' if quant_cfg else 'no'}")
                tok = AutoTokenizer.from_pretrained(candidate, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token

                mdl = AutoModelForCausalLM.from_pretrained(
                    candidate,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    quantization_config=quant_cfg,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False
                )

                with self._lock:
                    self.tokenizer = tok
                    self.model = mdl
                    self.model_id = candidate
                    if quant_cfg and use_gpu:
                        self.engine = f"{candidate} (4-bit GPU)"
                    elif use_gpu:
                        self.engine = f"{candidate} (GPU)"
                    else:
                        self.engine = f"{candidate} (CPU)"

                self.log(f"Loaded {self.engine}")
                self.loading = False
                return

            except Exception as e:
                self.log(f"Failed to load {candidate}: {e}")

        # If all models failed
        self.log("All candidate models failed → fallback to rule-based.")
        with self._lock:
            self.tokenizer = None
            self.model = None
            self.model_id = None
            self.engine = "rule-based"
        self.loading = False

    def reload(self):
        """Hot reload the model."""
        self.loading = True
        with self._lock:
            self.tokenizer = None
            self.model = None
            self.engine = "loading"
            self.model_id = None
        threading.Thread(target=self._load_best_model, daemon=True).start()

    def _format_memory(self, rows: List[Tuple[str,str,str]], limit=10) -> str:
        lines = []
        for (ts, v, a) in rows[:limit]:
            try:
                vj = json.loads(v) if v else {}
                aj = json.loads(a) if a else {}
            except Exception:
                vj, aj = v, a
            lines.append(f"- {ts} | Vision={vj} | Audio={aj}")
        return "\n".join(lines) if lines else "(no recent memory)"

    def _rule_based_answer(self, question: str) -> str:
        rows = self.memory.get_last_n(10)
        context = self.context.last_summary
        person_count = 0
        speaking_count = 0
        for _, v, a in rows:
            try:
                vj = json.loads(v) if v else {}
                aj = json.loads(a) if a else {}
            except Exception:
                vj, aj = {}, {}
            if vj.get("person_detected", False):
                person_count += 1
            if aj.get("speaking", False):
                speaking_count += 1

        q = question.lower()
        if "memory" in q:
            return f"I recorded {len(rows)} recent events. Notably, there were {person_count} frames with someone visible and {speaking_count} audio chunks indicating speech."
        if "status" in q or "now" in q:
            return f"Current context: {context}"
        if "summary" in q:
            return f"Summary: {context}. In the last {len(rows)} entries, visibility occurred {person_count} times and speech occurred {speaking_count} times."
        if "tip" in q or "advice" in q:
            if speaking_count > person_count:
                return "Tip: Since I detect more speech than faces, consider checking microphone sensitivity or ensure the camera has a clear view."
            elif person_count > speaking_count:
                return "Tip: Faces are detected often but speech is low—if you're testing voice, try speaking closer to the mic."
            else:
                return "Tip: Everything looks balanced. If you're demoing, try moving and speaking to show both modules working."
        return f"Here's what I know right now: {context}. Ask about 'memory', 'summary', or request a 'tip' for a more specific response."

    def _build_prompt(self, question: str) -> str:
        rows = self.memory.get_last_n(12)
        mem_text = self._format_memory(rows, limit=12)
        context = self.context.last_summary
        # Generic friendly prompt works across phi-1.5 and TinyLlama
        prompt = (
            "You are a small offline assistant inside a local AI Companion.\n"
            "Be concise and practical. Use the recent memory and the context to answer.\n\n"
            f"CONTEXT: {context}\n"
            f"RECENT MEMORY (most recent first):\n{mem_text}\n\n"
            f"USER: {question}\nASSISTANT:"
        )
        return prompt

    def _llm_answer(self, question: str) -> str:
        if self.loading:
            return "AI model is still loading; using quick rule-based answer for now.\n" + self._rule_based_answer(question)
        if self.model is None or self.tokenizer is None:
            return self._rule_based_answer(question)

        prompt = self._build_prompt(question)
        try:
            inputs = self.tokenizer([prompt], return_tensors="pt")
            # Do NOT forcibly .cuda() for device_map='auto' models; let HF handle placement.
            gen_cfg = dict(
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_cfg)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "ASSISTANT:" in text:
                return text.split("ASSISTANT:", 1)[1].strip()
            return text.strip()
        except Exception as e:
            self.log(f"Generation error: {e}")
            return self._rule_based_answer(question)

    def answer(self, question: str) -> Dict[str, str]:
        if self.model is not None and self.tokenizer is not None and not self.loading:
            return {"answer": self._llm_answer(question), "engine": self.engine}
        return {"answer": self._rule_based_answer(question), "engine": self.engine + (" (loading)" if self.loading else "")}

    def quick_tip(self) -> Dict[str, str]:
        q = "Give me a brief tip based on what you've seen and heard."
        return self.answer(q)

# =================== USER (Commands) ===================
class UserInteractionModule(BaseModule):
    def __init__(self, memory: MemoryModule):
        super().__init__("UserInteractionModule")
        self.memory = memory

    def process_command(self, cmd: str) -> str:
        cmd = cmd.strip().lower()
        if cmd == "show memory":
            rows = self.memory.get_last_n(5)
            resp = "[User] Recent memory entries:<br>"
            for r in rows:
                resp += f"{r}<br>"
            return resp
        elif cmd.startswith("summarize last"):
            try:
                minutes = int(cmd.split()[2])
            except:
                minutes = 5
            rows = self.memory.get_since(minutes)
            return f"[User] Context summary for last {minutes} minutes:<br>{len(rows)} entries recorded."
        elif cmd == "quit":
            # Note: We don't actually quit Flask here; user should Ctrl+C terminal
            return "[User] Quitting... (use Ctrl+C in terminal to stop app)"
        else:
            return "[User] Unknown command."

# =================== ORCHESTRATOR ===================
class AICompanion:
    def __init__(self):
        self.vision = VisionModule(camera_index="http://192.168.117.37:8080/video", show_window=SHOW_WINDOW)
        self.audio = AudioModule()
        self.memory = MemoryModule()
        self.context = ContextModule()
        self.user = UserInteractionModule(self.memory)
        self.reasoner = ReasoningModule(self.memory, self.context)
        self.modules = [self.vision, self.audio, self.memory, self.context, self.user, self.reasoner]
        self._running = False
        self.latest_ctx = {}

    def start(self):
        for m in self.modules:
            if hasattr(m, "start"):
                try:
                    m.start()
                except Exception as e:
                    print(f"[App] Failed to start {m.name}: {e}")
        self._running = True
        print("[App] Started", flush=True)

    def run(self):
        try:
            while self._running:
                ctx = {}
                v_out = self.vision.run_once(ctx) if self.vision else None
                a_out = self.audio.run_once(ctx) if self.audio else None
                if v_out:
                    ctx.update(v_out)
                if a_out:
                    ctx.update(a_out)
                self.memory.run_once(ctx)
                c_out = self.context.run_once(ctx)
                if c_out:
                    ctx.update(c_out)
                self.latest_ctx = ctx
                time.sleep(0.18)
        except KeyboardInterrupt:
            print("[App] Ctrl+C stopping...")
        finally:
            self.stop()

    def stop(self):
        for m in reversed(self.modules):
            if hasattr(m, "stop"):
                try:
                    m.stop()
                except Exception:
                    pass
        self._running = False
        print("[App] Stopped", flush=True)

# =================== FLASK APP ===================
flask_app = Flask(__name__)
companion = AICompanion()

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Companion Dashboard</title>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
      color: #ffffff;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Animated background particles */
    .bg-animation {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: -1;
    }

    .particle {
      position: absolute;
      background: rgba(59, 130, 246, 0.1);
      border-radius: 50%;
      animation: float 20s infinite linear;
    }

    @keyframes float {
      0% {
        transform: translateY(100vh) rotate(0deg);
        opacity: 0;
      }
      10% {
        opacity: 1;
      }
      90% {
        opacity: 1;
      }
      100% {
        transform: translateY(-100vh) rotate(360deg);
        opacity: 0;
      }
    }

    /* Header */
    .header {
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding: 20px 0;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .header-content {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 20px;
    }

    .logo {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(45deg, #3b82f6, #8b5cf6, #06b6d4);
      background-clip: text;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
    }

    .logo i {
      margin-right: 15px;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.1); }
    }

    /* Main container */
    .container {
      max-width: 1200px;
      margin: 40px auto;
      padding: 0 20px;
    }

    /* Grid layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 30px;
      margin-bottom: 30px;
    }

    .full-width {
      grid-column: span 2;
    }

    /* Glass card effect */
    .card {
      background: rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(15px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 20px;
      padding: 25px;
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }

    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    }

    .card:hover {
      transform: translateY(-5px);
      box-shadow: 0 20px 40px rgba(59, 130, 246, 0.2);
      border-color: rgba(59, 130, 246, 0.3);
    }

    .card h3 {
      font-size: 1.4rem;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      color: #ffffff;
    }

    .card h3 i {
      margin-right: 12px;
      color: #3b82f6;
      font-size: 1.2em;
    }

    /* Status indicators */
    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 15px;
      margin-bottom: 20px;
    }

    .status-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 16px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .status-label {
      font-weight: 600;
      color: #e5e7eb;
    }

    .status-indicator {
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 600;
      min-width: 80px;
      text-align: center;
      transition: all 0.3s ease;
    }

    .status-ok {
      background: linear-gradient(135deg, #10b981, #059669);
      color: white;
      box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }

    .status-error {
      background: linear-gradient(135deg, #ef4444, #dc2626);
      color: white;
      box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }

    .status-neutral {
      background: linear-gradient(135deg, #6b7280, #4b5563);
      color: white;
      box-shadow: 0 4px 15px rgba(107, 114, 128, 0.3);
    }

    .status-loading {
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: white;
      box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
      animation: loading-pulse 1.5s infinite;
    }

    @keyframes loading-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; }
    }

    /* Buttons */
    .button-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .btn {
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      color: white;
      border: none;
      padding: 12px 20px;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
      font-size: 0.9rem;
      position: relative;
      overflow: hidden;
    }

    .btn::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
      transition: left 0.5s;
    }

    .btn:hover::before {
      left: 100%;
    }

    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
    }

    .btn-secondary {
      background: linear-gradient(135deg, #6b7280, #4b5563);
    }

    .btn-secondary:hover {
      background: linear-gradient(135deg, #4b5563, #374151);
      box-shadow: 0 10px 25px rgba(107, 114, 128, 0.4);
    }

    .btn-accent {
      background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    }

    .btn-accent:hover {
      background: linear-gradient(135deg, #7c3aed, #6d28d9);
      box-shadow: 0 10px 25px rgba(139, 92, 246, 0.4);
    }

    /* Camera section */
    .camera-container {
      text-align: center;
      position: relative;
    }

    .camera-frame {
      border-radius: 16px;
      overflow: hidden;
      position: relative;
      margin-bottom: 15px;
      background: rgba(255, 255, 255, 0.05);
      border: 2px solid rgba(59, 130, 246, 0.3);
    }

    .camera-frame img {
      width: 100%;
      height: auto;
      max-height: 300px;
      object-fit: cover;
      display: block;
    }

    .camera-status {
      position: absolute;
      top: 15px;
      right: 15px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 8px 12px;
      border-radius: 20px;
      font-size: 0.8rem;
      backdrop-filter: blur(10px);
    }

    /* AI Chat section */
    .chat-container {
      margin-bottom: 20px;
    }

    .input-group {
      display: flex;
      gap: 12px;
      margin-bottom: 20px;
    }

    .chat-input {
      flex: 1;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 12px;
      padding: 15px 20px;
      color: white;
      font-size: 1rem;
      transition: all 0.3s ease;
    }

    .chat-input:focus {
      outline: none;
      border-color: #3b82f6;
      background: rgba(255, 255, 255, 0.15);
      box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }

    .chat-input::placeholder {
      color: rgba(255, 255, 255, 0.6);
    }

    .response-container {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      padding: 20px;
      min-height: 120px;
      font-family: 'Consolas', 'Monaco', monospace;
      font-size: 0.9rem;
      line-height: 1.6;
      white-space: pre-wrap;
      position: relative;
      overflow-y: auto;
      max-height: 300px;
    }

    .response-container:empty::before {
      content: 'AI responses will appear here...';
      color: rgba(255, 255, 255, 0.5);
      font-style: italic;
    }

    /* Quick actions */
    .quick-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 15px;
    }

    .quick-btn {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 8px 16px;
      border-radius: 20px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.3s ease;
    }

    .quick-btn:hover {
      background: rgba(59, 130, 246, 0.3);
      border-color: rgba(59, 130, 246, 0.5);
    }

    /* Responsive design */
    @media (max-width: 768px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
      
      .full-width {
        grid-column: span 1;
      }

      .status-grid,
      .button-grid {
        grid-template-columns: 1fr;
      }

      .input-group {
        flex-direction: column;
      }

      .quick-actions {
        justify-content: center;
      }

      .logo {
        font-size: 2rem;
      }
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
      width: 8px;
    }

    ::-webkit-scrollbar-track {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
      background: rgba(59, 130, 246, 0.6);
      border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
      background: rgba(59, 130, 246, 0.8);
    }

    /* Loading animation */
    .loading {
      display: inline-block;
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #3b82f6;
      animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <!-- Animated background -->
  <div class="bg-animation" id="bgAnimation"></div>

  <!-- Header -->
  <header class="header">
    <div class="header-content">
      <h1 class="logo">
        <i class="fas fa-robot"></i>
        AI Companion Dashboard
      </h1>
    </div>
  </header>

  <!-- Main content -->
  <div class="container">
    <div class="dashboard-grid">
      <!-- System Status Card -->
      <div class="card">
        <h3><i class="fas fa-heartbeat"></i>System Status</h3>
        <div class="status-grid">
          <div class="status-item">
            <span class="status-label"><i class="fas fa-eye"></i> Vision</span>
            <span id="vision" class="status-indicator status-neutral">Initializing</span>
          </div>
          <div class="status-item">
            <span class="status-label"><i class="fas fa-microphone"></i> Audio</span>
            <span id="audio" class="status-indicator status-neutral">Initializing</span>
          </div>
          <div class="status-item">
            <span class="status-label"><i class="fas fa-brain"></i> Context</span>
            <span id="context" class="status-indicator status-neutral">Ready</span>
          </div>
          <div class="status-item">
            <span class="status-label"><i class="fas fa-cog"></i> AI Engine</span>
            <span id="ai" class="status-indicator status-loading">Loading</span>
          </div>
        </div>
        <div class="button-grid">
          <button class="btn" onclick="reloadModel()">
            <i class="fas fa-sync-alt"></i> Reload Model
          </button>
          <button class="btn btn-secondary" onclick="sendCommand('show memory')">
            <i class="fas fa-database"></i> Show Memory
          </button>
          <button class="btn btn-secondary" onclick="sendCommand('summarize last 5')">
            <i class="fas fa-chart-line"></i> Summarize
          </button>
        </div>
      </div>

      <!-- Camera Feed Card -->
      <div class="card">
        <h3><i class="fas fa-video"></i>Live Camera Feed</h3>
        <div class="camera-container">
          <div class="camera-frame">
            <img id="cam" src="/video_feed" alt="Camera initializing..." />
            <div class="camera-status">
              <i class="fas fa-circle text-green-400"></i> Live
            </div>
          </div>
          <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.85rem;">
            Real-time vision processing with face detection
          </p>
        </div>
      </div>

      <!-- AI Chat Interface Card -->
      <div class="card full-width">
        <h3><i class="fas fa-comments"></i>AI Assistant</h3>
        <div class="chat-container">
          <div class="input-group">
            <input 
              id="ask" 
              type="text" 
              class="chat-input" 
              placeholder="Ask me anything about what I've seen or heard..."
              onkeypress="if(event.key==='Enter') askAI()"
            >
            <button class="btn" onclick="askAI()">
              <i class="fas fa-paper-plane"></i> Ask
            </button>
            <button class="btn btn-accent" onclick="quickTip()">
              <i class="fas fa-lightbulb"></i> Quick Tip
            </button>
          </div>
          
          <div class="response-container" id="response">{{response}}</div>
          
          <div class="quick-actions">
            <button class="quick-btn" onclick="askQuick('What happened recently?')">
              <i class="fas fa-clock"></i> Recent Activity
            </button>
            <button class="quick-btn" onclick="askQuick('Any advice for me?')">
              <i class="fas fa-question-circle"></i> Get Advice
            </button>
            <button class="quick-btn" onclick="askQuick('What do you see right now?')">
              <i class="fas fa-search"></i> Current Status
            </button>
            <button class="quick-btn" onclick="askQuick('How are the sensors?')">
              <i class="fas fa-sensors"></i> Sensor Check
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Create animated background particles
    function createParticles() {
      const container = document.getElementById('bgAnimation');
      const particleCount = 20;
      
      for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.width = particle.style.height = Math.random() * 4 + 2 + 'px';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
        container.appendChild(particle);
      }
    }

    // Status update functionality
    async function updateStatus() {
      try {
        const res = await fetch("/status");
        const data = await res.json();
        setStatus("vision", data.vision);
        setStatus("audio", data.audio);
        setContextStatus(data.context);
        setAIStatus(data.ai);
      } catch (error) {
        console.error('Status update failed:', error);
      }
    }

    function setStatus(id, text) {
      const el = document.getElementById(id);
      el.innerText = getStatusText(text);
      el.className = "status-indicator " + getStatusClass(text);
    }

    function setContextStatus(text) {
      const el = document.getElementById("context");
      el.innerText = "Active";
      el.className = "status-indicator status-ok";
      el.title = text;
    }

    function setAIStatus(text) {
      const el = document.getElementById("ai");
      el.innerText = text;
      
      if (text.toLowerCase().includes("loading")) {
        el.className = "status-indicator status-loading";
      } else if (text.toLowerCase().includes("phi") || text.toLowerCase().includes("tinyllama")) {
        el.className = "status-indicator status-ok";
      } else if (text.toLowerCase().includes("rule-based")) {
        el.className = "status-indicator status-neutral";
      } else {
        el.className = "status-indicator status-neutral";
      }
    }

    function getStatusText(text) {
      if (text.includes("Person detected") || text.includes("✅")) return "Active";
      if (text.includes("Speaking") || text.includes("🎤")) return "Speaking";
      if (text.includes("No person") || text.includes("❌")) return "Standby";
      if (text.includes("Silent") || text.includes("🤐")) return "Silent";
      return "Ready";
    }

    function getStatusClass(text) {
      if (text.includes("✅") || text.includes("🎤") || text.includes("Person detected") || text.includes("Speaking")) {
        return "status-ok";
      }
      if (text.includes("❌") || text.includes("🤐") || text.includes("fail")) {
        return "status-error";
      }
      return "status-neutral";
    }

    // Command functionality
    async function sendCommand(cmd) {
      try {
        const res = await fetch("/command", {
          method: "POST",
          headers: {"Content-Type": "application/x-www-form-urlencoded"},
          body: "cmd=" + encodeURIComponent(cmd)
        });
        const html = await res.text();
        // Instead of replacing the entire document, show result in response box
        const responseBox = document.getElementById("response");
        responseBox.textContent = `[Command: ${cmd}]\nExecuted successfully. Check console for details.`;
      } catch (error) {
        console.error('Command failed:', error);
        const responseBox = document.getElementById("response");
        responseBox.textContent = `[Error]\nCommand failed: ${error.message}`;
      }
    }

    // AI interaction functionality
    async function askAI() {
      const q = document.getElementById("ask").value.trim();
      if (!q) return;
      
      const responseBox = document.getElementById("response");
      responseBox.innerHTML = '<div class="loading"></div> Processing your question...';
      
      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question: q})
        });
        const data = await res.json();
        responseBox.textContent = `[Engine: ${data.engine}]\n\n${data.answer}`;
        document.getElementById("ask").value = '';
      } catch (error) {
        responseBox.textContent = `[Error]\nFailed to get response: ${error.message}`;
      }
    }

    async function askQuick(question) {
      document.getElementById("ask").value = question;
      await askAI();
    }

    async function quickTip() {
      const responseBox = document.getElementById("response");
      responseBox.innerHTML = '<div class="loading"></div> Generating tip...';
      
      try {
        const res = await fetch("/tip", { method: "POST" });
        const data = await res.json();
        responseBox.textContent = `[Engine: ${data.engine}]\n\n💡 ${data.answer}`;
      } catch (error) {
        responseBox.textContent = `[Error]\nFailed to get tip: ${error.message}`;
      }
    }

    async function reloadModel() {
      const responseBox = document.getElementById("response");
      responseBox.innerHTML = '<div class="loading"></div> Reloading AI model...';
      
      try {
        const res = await fetch("/reload", { method: "POST" });
        const data = await res.json();
        responseBox.textContent = `[Engine: ${data.engine}]\n\n${data.message}`;
      } catch (error) {
        responseBox.textContent = `[Error]\nFailed to reload model: ${error.message}`;
      }
    }

    // Initialize
    document.addEventListener('DOMContentLoaded', function() {
      createParticles();
      updateStatus();
      setInterval(updateStatus, 2000);
      
      // Welcome message
      setTimeout(() => {
        const responseBox = document.getElementById("response");
        if (!responseBox.textContent.trim()) {
          responseBox.textContent = "🤖 AI Companion is ready!\\n\\nTry asking me something like:\\n• 'What do you see right now?'\\n• 'Any recent activity?'\\n• 'Give me some advice'\\n\\nI can analyze vision, audio, and provide insights based on my observations.";
        }
      }, 1000);
    });

    // Handle image load errors
    document.getElementById('cam').onerror = function() {
      this.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhbWVyYSBOb3QgQXZhaWxhYmxlPC90ZXh0Pjwvc3ZnPg==';
    };
  </script>
</body>
</html>
"""

# ============== Routes ==============

@flask_app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, response="")

@flask_app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = companion.vision.get_latest_frame() if companion.vision else None
            if frame is None:
                # serve a gray image
                img = np.zeros((240, 320, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', img)
            else:
                # draw faces
                faces = []
                # we already draw in the opencv window, but for streaming just pass raw for speed
                _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@flask_app.route("/command", methods=["POST"])
def handle_command():
    cmd = request.form.get("cmd", "")
    response = companion.user.process_command(cmd)
    return render_template_string(HTML_TEMPLATE, response=response)

@flask_app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please ask a question.", "engine": "n/a"})
    out = companion.reasoner.answer(question)
    return jsonify(out)

@flask_app.route("/tip", methods=["POST"])
def tip():
    out = companion.reasoner.quick_tip()
    return jsonify(out)

@flask_app.route("/reload", methods=["POST"])
def reload_model():
    companion.reasoner.reload()
    return jsonify({"message": "Reloading model… You can continue using the app.", "engine": "loading"})

@flask_app.route("/status")
def status():
    ctx = companion.latest_ctx or {}
    vision = "Person detected ✅" if ctx.get("vision", {}).get("person_detected") else "No person ❌"
    audio = "Speaking 🎤" if ctx.get("audio", {}).get("speaking") else "Silent 🤐"
    context = companion.context.last_summary
    ai = companion.reasoner.engine + (" (loading...)" if companion.reasoner.loading else "")
    return jsonify({"vision": vision, "audio": audio, "context": context, "ai": ai})

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# ============== Main ==============
if __name__ == "__main__":
    # Friendly banner
    print("⏳ Starting AI Companion…", flush=True)
    if not TRANSFORMERS_OK:
        print("⚠  transformers not available — will run in rule-based mode unless installed.", flush=True)
    else:
        print(f"🔎 Will try models in order: {MODEL_CANDIDATES}", flush=True)
        print(f"   Device: {'GPU' if CUDA_OK and not FORCE_CPU else 'CPU'} | 4-bit: {'ON' if (LOAD_IN_4BIT and BITSANDBYTES_OK and CUDA_OK) else 'OFF'}", flush=True)

    companion.start()
    # Run background AI loop
    t = threading.Thread(target=companion.run, daemon=True)
    t.start()

    ip = get_local_ip()
    print(f"[Flask] Web UI running → http://127.0.0.1:5000  OR  http://{ip}:5000", flush=True)
    flask_app.run(host="0.0.0.0", port=5000, debug=False)