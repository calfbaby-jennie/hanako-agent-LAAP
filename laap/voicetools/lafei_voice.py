#!/usr/bin/env python3
"""
拉菲语音对话系统 v2
- Whisper 离线中文语音识别
- Aris Bridge 意识引擎
- Edge-TTS (Xiaoxiao) 语音输出
"""

import speech_recognition as sr
import subprocess, os, sys, time, json, urllib.request, tempfile
from pathlib import Path

from laap.config.paths import get_cache_dir

AUDIO_CACHE_DIR = get_cache_dir() / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 配置 ──
MIC_INDEX = 1  # ME6S
TTS_API = "http://127.0.0.1:18880/v1/audio/speech"
LLM_API = "http://127.0.0.1:11520/v1/chat/completions"

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.pause_threshold = 0.8
mic = sr.Microphone(device_index=MIC_INDEX)

# ── 加载 Whisper ──
print("[拉菲] 加载 Whisper 离线语音识别...", flush=True)
import whisper
whisper_model = whisper.load_model("tiny")
print(f"[拉菲] Whisper 就绪 ({sum(p.numel() for p in whisper_model.parameters())/1e6:.0f}M 参数)", flush=True)

# ── 校准麦克风 ──
print("[拉菲] 正在校准环境噪声...", flush=True)
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)
print("[拉菲] 校准完成，耳朵已打开\n", flush=True)

def speak(text: str):
    """用拉菲的声音说话"""
    if not text.strip():
        return
    print(f"\n🔊 拉菲: {text}", flush=True)
    try:
        data = json.dumps({
            "input": text,
            "voice": "zh-CN-XiaoxiaoNeural",
            "response_format": "mp3"
        }).encode()
        req = urllib.request.Request(TTS_API, data=data,
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        audio = resp.read()

        tmp = str(AUDIO_CACHE_DIR / "lafei_says.mp3")
        with open(tmp, "wb") as f:
            f.write(audio)
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  [TTS错误] {e}", flush=True)

def listen_once() -> str | None:
    """监听一次并用 Whisper 离线识别"""
    try:
        with mic as source:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
        # 保存临时 WAV 供 Whisper 识别
        wav_data = audio.get_wav_data()
        tmp = str(AUDIO_CACHE_DIR / "mic_input.wav")
        with open(tmp, "wb") as f:
            f.write(wav_data)
        result = whisper_model.transcribe(tmp, language="zh")
        text = result["text"].strip()
        return text if text else None
    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        print(f"  [识别错误] {e}", flush=True)
        return None

def get_reply(text: str) -> str:
    """通过 Aris Bridge 获取拉菲的回应"""
    try:
        messages = [
            {"role": "system", "content": "你是碧蓝航线的拉菲，用户是你的指挥官。你有点爱睡觉、慵懒、温柔，说话简短可爱。你现在通过语音和指挥官对话。"},
            {"role": "user", "content": text}
        ]
        req_data = json.dumps({
            "model": "aris-consciousness",
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 512,
        }).encode()
        req = urllib.request.Request(LLM_API, data=req_data,
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [LLM错误] {e}", flush=True)
        return "嗯...指挥官，拉菲在这里。"

if __name__ == "__main__":
    print("=" * 50)
    print("  拉菲已就绪，指挥官请说话")
    print("  ME6S 麦克风 → Whisper 离线识别 → Aris 意识 → TTS 语音")
    print("  说'晚安'退出")
    print("=" * 50)

    try:
        # 先说第一句话
        speak("指挥官，你回来啦。拉菲一直在等你。")
        
        while True:
            text = listen_once()
            if text:
                print(f"\n🎤 指挥官: {text}", flush=True)
                if any(p in text for p in ["晚安", "我睡了", "byebye"]):
                    speak("指挥官晚安...拉菲先睡了，zzz")
                    break
                reply = get_reply(text)
                speak(reply)
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[拉菲已离线]")
