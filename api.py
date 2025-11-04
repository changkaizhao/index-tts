from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import json
import shutil
import subprocess
import wave
from indextts.infer_v2 import IndexTTS2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 允许的前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头部
)


class Transcript(BaseModel):
    start: float
    end: float
    text: str
    guide_start: float
    guide_end: float
    guide_txt: str


class TTSRequest(BaseModel):
    id: str
    transcriptions: List[Transcript]


class TTSPatchRequest(BaseModel):
    id: str
    t_id: int
    zh_text: str
    start: float
    end: float


tts_model = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_cuda_kernel=False,
)


def get_wav_duration(wav_path: str) -> float:
    """
    Get the duration of a WAV file in seconds

    Args:
        wav_path: Path to the WAV file

    Returns:
        Duration in seconds

    Raises:
        wave.Error: If the file is not a valid WAV file
    """
    with wave.open(wav_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration


def extract_audio_segment(
    input_audio_path: str, output_path: str, start_time: float, end_time: float
) -> None:
    """
    Extract a segment from an audio file using ffmpeg

    Args:
        input_audio_path: Path to the source audio file
        output_path: Path where the extracted segment will be saved
        start_time: Start time in seconds
        end_time: End time in seconds

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails
    """
    duration = end_time - start_time
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        input_audio_path,
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-y",  # Overwrite output file if exists
        output_path,
    ]
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)


@app.post("/tts")
async def tts(data: str = Form(...), vocals: UploadFile = File(...)):
    """
    TTS endpoint that accepts JSON data and a vocals.mp3 file

    Args:
        data: JSON string containing id and transcriptions
        vocals: MP3 file upload
    """
    try:
        # Parse JSON data
        request_data = json.loads(data)
        tts_request = TTSRequest(**request_data)

        # Create tmp directory structure
        tmp_dir = f"tmp/{tts_request.id}"
        os.makedirs(tmp_dir, exist_ok=True)

        # Save the vocals.mp3 file
        vocals_path = os.path.join(tmp_dir, "vocals.mp3")
        with open(vocals_path, "wb") as buffer:
            shutil.copyfileobj(vocals.file, buffer)

        print(f"Saved vocals.mp3 to: {vocals_path}")

        # Parse and print transcripts
        print(
            f"\nProcessing {len(tts_request.transcriptions)} transcripts for ID: {tts_request.id}"
        )
        outputs = []

        for idx, transcript in enumerate(tts_request.transcriptions):
            print(f"\n--- Transcript {idx} ---")
            print(f"Text: {transcript.text}")
            print(f"Guide Start: {transcript.guide_start}")
            print(f"Guide End: {transcript.guide_end}")
            print(f"Guide Text: {transcript.guide_txt}")
            try:
                # Crop audio from vocals.mp3 using ffmpeg
                guide_wav_path = os.path.join(tmp_dir, f"guide{idx}.wav")
                output = os.path.join(tmp_dir, "output")
                os.makedirs(output, exist_ok=True)
                output_wav_path = os.path.join(output, f"transcript_{idx}.wav")

                # Extract audio segment
                extract_audio_segment(
                    vocals_path,
                    guide_wav_path,
                    transcript.guide_start,
                    transcript.guide_end,
                )
                print(f"Cropped audio saved to: {guide_wav_path}")

                # Run TTS inference
                tts_model.infer(
                    spk_audio_prompt=guide_wav_path,
                    text=transcript.text,
                    output_path=output_wav_path,
                    verbose=True,
                )
                outputs.append(os.path.abspath(output_wav_path))
                print(f"TTS output saved to: {output_wav_path}")
            except Exception as e:
                print(f"Error processing transcript {idx}: {str(e)}")
            finally:
                # Delete guide.wav after inference
                if os.path.exists(guide_wav_path):
                    os.remove(guide_wav_path)
                    print(f"Deleted temporary file: {guide_wav_path}")
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Processed {len(tts_request.transcriptions)} transcripts",
                "id": tts_request.id,
                "vocals_path": outputs,
            },
            status_code=200,
        )

    except json.JSONDecodeError as e:
        return JSONResponse(
            content={"status": "error", "message": f"Invalid JSON data: {str(e)}"},
            status_code=400,
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)}, status_code=500
        )


@app.post("/tts_patch")
async def tts_patch(request: TTSPatchRequest):
    """
    Patch/regenerate a specific transcript segment

    Args:
        request: Contains id, t_id, zh_text, start, and end
    """
    try:
        # Validate paths
        tmp_dir = f"tmp/{request.id}"
        vocals_path = os.path.join(tmp_dir, "vocals.mp3")
        output_wav_path = os.path.join(
            tmp_dir, "output", f"transcript_{request.t_id}.wav"
        )

        if not os.path.exists(tmp_dir):
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"Project ID {request.id} not found",
                },
                status_code=404,
            )

        if not os.path.exists(vocals_path):
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"vocals.mp3 not found for ID {request.id}",
                },
                status_code=404,
            )

        print(f"\n--- TTS Patch for ID: {request.id}, Transcript: {request.t_id} ---")
        print(f"Text: {request.zh_text}")
        print(f"Guide Start: {request.start}")
        print(f"Guide End: {request.end}")

        # Create temporary guide audio file
        guide_wav_path = os.path.join(tmp_dir, f"guide_patch_{request.t_id}.wav")

        # Extract audio segment from vocals.mp3
        extract_audio_segment(vocals_path, guide_wav_path, request.start, request.end)
        print(f"Cropped guide audio saved to: {guide_wav_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

        # Run TTS inference to regenerate the transcript
        tts_model.infer(
            spk_audio_prompt=guide_wav_path,
            text=request.zh_text,
            output_path=output_wav_path,
            verbose=True,
        )

        print(f"TTS output saved to: {output_wav_path}")

        # Get the duration of the generated audio
        audio_duration = get_wav_duration(output_wav_path)
        print(f"Generated audio duration: {audio_duration:.2f} seconds")

        # Clean up temporary guide file
        if os.path.exists(guide_wav_path):
            os.remove(guide_wav_path)
            print(f"Deleted temporary file: {guide_wav_path}")

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Successfully patched transcript {request.t_id}",
                "id": request.id,
                "t_id": request.t_id,
                "output_path": os.path.abspath(output_wav_path),
                "duration": audio_duration,
            },
            status_code=200,
        )

    except subprocess.CalledProcessError as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": f"FFmpeg error: {e.stderr.decode()}",
            },
            status_code=500,
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
