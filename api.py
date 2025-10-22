from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import json
import shutil
import subprocess
from indextts.infer_v2 import IndexTTS2

app = FastAPI()


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


tts_model = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_cuda_kernel=False,
)


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
                guide_wav_path = os.path.join(tmp_dir, "guide.wav")
                duration = transcript.guide_end - transcript.guide_start

                # Use ffmpeg to extract audio segment
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    vocals_path,
                    "-ss",
                    str(transcript.guide_start),
                    "-t",
                    str(duration),
                    "-y",  # Overwrite output file if exists
                    guide_wav_path,
                ]
                output = os.path.join(tmp_dir, "output")
                os.makedirs(output, exist_ok=True)
                output_wav_path = os.path.join(output, f"transcript_{idx}.wav")

                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
