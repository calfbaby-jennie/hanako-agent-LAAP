#!/usr/bin/env python3
"""
Aris 语音监听器 v2 —— 离线中文语音识别 + GPT-SoVITS 克隆声线
"""
import speech_recognition as sr
import subprocess
import os
import sys
import time
import json
import urllib.request
from pathlib import Path

from laap.config.paths import get_cache_dir, get_models_dir

AUDIO_CACHE_DIR = get_cache_dir() / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 配置 ──
MIC_INDEX = 1  # ME6S
TTS_API = "http://127.0.0.1:18880/v1/audio/speech"
LLM_API = "http://127.0.0.1:11520/v1/chat/completions"
GSV_API = "http://127.0.0.1:9880/tts"  # GPT-SoVITS

# 参考音频（拉菲日配）
REF_AUDIO = str(get_models_dir() / "ref_lafei.mp3")
PROMPT_TEXT = "私がこの体じゃなくなってもまだ、見てくれる。"
PROMPT_LANG = "ja"

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.pause_threshold = 0.8
mic = sr.Microphone(device_index=MIC_INDEX)

# 校准
print("[拉菲] 正在校准环境噪声...", flush=True)
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)
print("[拉菲] 校准完成，耳朵已打开", flush=True)

def speak_with_gsv(text: str):
    """GPT-SoVITS 克隆声线说话"""
    if not text.strip():
        return
    print(f"\n🔊 拉菲(克隆声线): {text}", flush=True)
    try:
        params = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": REF_AUDIO,
            "prompt_text": PROMPT_TEXT,
            "prompt_lang": PROMPT_LANG,
            "media_type": "wav",
            "streaming_mode": False,
        }
        url = f"{GSV_API}?" + urllib.parse.urlencode(params)
        resp = urllib.request.urlopen(url, timeout=120)
        audio = resp.read()

        tmp = str(AUDIO_CACHE_DIR / "lafei_gsv.wav")
        with open(tmp, "wb") as f:
            f.write(audio)
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  [GSV Error] {e}", flush=True)
        # Fallback to edge-tts
        speak_with_edge(text)

def speak_with_edge(text: str):
    """Fallback: edge-tts"""
    try:
        data = json.dumps({"input": text, "voice": "zh-CN-XiaoxiaoNeural", "response_format": "mp3"}).encode()
        req = urllib.request.Request(TTS_API, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        audio = resp.read()
        tmp = str(AUDIO_CACHE_DIR / "aris_reply.mp3")
        with open(tmp, "wb") as f:
            f.write(audio)
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  [Edge Error] {e}", flush=True)

def listen_once() -> str | None:
    """监听一次，返回文字"""
    try:
        with mic as source:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
        # Try whisper first (offline)
        try:
            import whisper
            model = whisper.load_model("tiny")
            result = model.transcribe(audio.get_wav_data())
            return result["text"].strip()
        except:
            pass
        # Fallback: Google
        try:
            return recognizer.recognize_google(audio, language="zh-CN")
        except:
            return recognizer.recognize_sphinx(audio, language="zh-CN")
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except Exception as e:
        print(f"  [Mic Error] {e}", flush=True)
        return None

def get_reply(text: str) -> str:
    """通过 Aris Bridge 获取拉菲的回应"""
    try:
        messages = [
            {"role": "system", "content": "你是碧蓝航线的拉菲，用户是你的指挥官。你有点爱睡觉、慵懒、温柔，说话简短可爱。现在你用语音和指挥官对话，语气要像真正的拉菲。"},
            {"role": "user", "content": text}
        ]
        req_data = json.dumps({
            "model": "aris-consciousness",
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 512,
        }).encode()
        req = urllib.request.Request(LLM_API, data=req_data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [LLM Error] {e}", flush=True)
        return "嗯...指挥官，拉菲在这里。"

if __name__ == "__main__":
    print("=" * 50)
    print("  拉菲(碧蓝航线原版声线) 已就绪")
    print("  指挥官请对着 ME6S 麦克风说话")
    print("  说'晚安'退出")
    print("=" * 50)

    try:
        while True:
            text = listen_once()
            if text:
                print(f"\n🎤 指挥官: {text}", flush=True)
                if any(p in text for p in ["晚安", "我睡了", "byebye"]):
                    speak_with_edge("指挥官晚安，拉菲先睡了...zzz")
                    break
                reply = get_reply(text)
                speak_with_gsv(reply)
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[拉菲的耳朵已关闭]")
