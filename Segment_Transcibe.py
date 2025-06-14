"""
Split 12 anthem_* files (~60 s, on silence) and transcribe them.
Creates anthem_transcriptions.json in the same directory.
"""

import os, json, tempfile, shutil, time
from typing import List
import tqdm
from pydub import AudioSegment
from pydub.silence import detect_silence
import speech_recognition as sr


# -----------------  general parameters  -----------------

TARGET_CHUNK_MS      = 40_000          # ≈ 40-second chunks
SEARCH_WINDOW_MS     = 10_000          # look ±10s for a pause
MIN_SILENCE_LEN_MS   = 1_000           # pause must be ≥ 1s
EXTRA_SILENCE_MS     = 250             # pad so cuts sound natural
SILENCE_THRESH_PAD   = 15              # silence_thresh = dBFS – this value


# -----------------  helper functions  -------------------

def find_boundaries(audio: AudioSegment) -> List[int]:
    """
    Return list of millisecond offsets where to cut the audio.
    First boundary is 0, last one is len(audio).
    Cuts are placed at pauses as close as possible to multiples of TARGET_CHUNK_MS.
    """
    silence_list = detect_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=audio.dBFS - SILENCE_THRESH_PAD,
        seek_step=10
    )
    # keep only the *start* of each silence region
    silence_starts = [start for start, _ in silence_list]

    boundaries = [0]
    target = TARGET_CHUNK_MS
    length = len(audio)

    while target < length:
        # 1. pick silence inside [target-SEARCH_WINDOW, target+SEARCH_WINDOW]
        candidate = [s for s in silence_starts
                     if target - SEARCH_WINDOW_MS <= s <= target + SEARCH_WINDOW_MS]
        if candidate:
            cut = candidate[0]
        else:
            # 2. pick first silence *after* target
            later = [s for s in silence_starts if s > target]
            cut = later[0] if later else None

        if cut is None or cut - boundaries[-1] < 5_000:   # last chunk too small → finish
            break
        boundaries.append(cut)
        target = cut + TARGET_CHUNK_MS

    boundaries.append(length)
    return boundaries

def transcribe_chunk(chunk_path: str, recognizer: sr.Recognizer) -> str:
    """
    One-shot transcription. Returns empty string on failure.
    """
    with sr.AudioFile(chunk_path) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data, language='en-US')
    except (sr.UnknownValueError, sr.RequestError):
        return None


# -----------------  main processing  --------------------

def process_file(filepath: str, chapter_id: str) -> dict:
    print(f"-> {os.path.basename(filepath)}")
    audio = AudioSegment.from_file(filepath).set_channels(1).set_frame_rate(16_000)
    boundaries = find_boundaries(audio)

    r = sr.Recognizer()
    segments = []
    total_words = 0

    # create a temp directory for wav chunks
    tmpdir = tempfile.mkdtemp(prefix="chunks_")

    try:
        for idx in range(len(boundaries)-1):
            start_ms, end_ms = boundaries[idx], boundaries[idx+1]
            segment_audio = (AudioSegment.silent(EXTRA_SILENCE_MS) +
                             audio[start_ms:end_ms] +
                             AudioSegment.silent(EXTRA_SILENCE_MS))

            wav_path = os.path.join(tmpdir, f"{idx}.wav")
            segment_audio.export(wav_path, format="wav")

            text = transcribe_chunk(wav_path, r)
            if text is None:
                print(f"  Segment {idx:02d} transcription failed")
                continue
            word_count = len(text.split())

            segments.append(
                {
                    "segment_index": idx,
                    "starting_word_index": total_words,
                    "starting_time_seconds": round(start_ms / 1000.0, 2),
                    "text": text
                }
            )

            print(f"  Segment {idx:02d}: {word_count} words, starts at {(start_ms/1000):.2f}s")

            total_words += word_count
            os.remove(wav_path) # tidy up after every chunk
            time.sleep(0.05) # cheap throttle to avoid rate-limit

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "filename": os.path.basename(filepath),
        "total_duration_seconds": round(len(audio) / 1000.0, 2),
        "total_segments": len(segments),
        "total_words": total_words,
        "segments": segments
    }

def batch_segment_audio_from_json(json_path, audio_folder, output_folder):
    # Load JSON once
    with open(json_path, "r") as f:
        metadata = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    # Process each audio file in the folder
    for filename in os.listdir(audio_folder):
        if not filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
            continue

        audio_path = os.path.join(audio_folder, filename)

        # Find corresponding metadata entry
        chapter = next((v for v in metadata.values() if v.get("filename") == filename), None)

        if chapter is None:
            print(f"No metadata found for {filename}. Skipping.")
            continue

        total_duration = chapter.get("total_duration_seconds")
        segments = chapter.get("segments", [])

        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        for i, segment in enumerate(tqdm(segments, desc=f"Segmenting {filename}")):
            try:
                start_sec = segment["starting_time_seconds"]
                end_sec = (
                    segments[i + 1]["starting_time_seconds"]
                    if i + 1 < len(segments)
                    else total_duration
                )

                if end_sec <= start_sec:
                    print(f"Invalid segment {i} in {filename}. Skipping.")
                    continue

                start_ms = int(start_sec * 1000)
                end_ms = int(end_sec * 1000)

                segment_audio = audio[start_ms:end_ms]

                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_segment{i}.wav"
                output_path = os.path.join(output_folder, output_filename)

                segment_audio.export(output_path, format="wav")

                print(f"Saved {output_filename} ({start_sec:.2f}s → {end_sec:.2f}s)")

            except Exception as e:
                print(f"Error segmenting {filename} (segment {i}): {e}")

    print(f"✅ Done: all segments saved to {output_folder}")


output_folder = "\\Anthem_Segment_Audio"


def main():

    directory = '\\anthem_1308_librivox'

    if not os.path.isdir(directory):
        print("  Not a directory")
        return

    result = {}
    for n in range(1, 13):
        fname = f"anthem_{n:02d}_rand_64kb.mp3"
        fpath = os.path.join(directory, fname)
        if os.path.exists(fpath):
            chapter_key = f"chapter_{n:02d}"
            result[chapter_key] = process_file(fpath, chapter_key)
        else:
            print(f"  {fname} not found – skipping")

    out_path = os.path.join(directory, "anthem_transcriptions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  Finished. JSON written to {out_path}")

    batch_segment_audio_from_json(
        json_path="\\anthem_transcriptions.json",
        audio_folder="\\anthem_1308_librivox",
        output_folder="\\Anthem_Segment_Audio"
    )
if __name__ == "__main__":
    main()
