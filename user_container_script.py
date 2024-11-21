import os
import logging
import uvicorn
from fastapi import FastAPI, UploadFile, File, Response, status
from http import HTTPStatus
import uuid
import shutil
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = FastAPI()

@server.get("/")
def healthcheck():
    return HTTPStatus.OK

@server.post("/inference")
async def inference(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    temp_filepath = os.path.join('/tmp', temp_filename)
    with open(temp_filepath, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Build the command to run inference
    engine_dir = os.getenv('OUTPUT_DIR', './whisper_large_v3_int8')
    assets_dir = os.getenv('ASSETS_DIR', './assets')
    cmd = [
        'python3', 'run.py',
        '--name', 'single_wav_test',
        '--engine_dir', engine_dir,
        '--input_file', temp_filepath,
        '--assets_dir', assets_dir,
        '--padding_strategy', 'max'  # Adjust based on your preference
    ]

    # Run the command and capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Delete the temporary audio file
    os.remove(temp_filepath)

    # Check if the command was successful
    if result.returncode != 0:
        logger.error(f"Inference failed: {result.stderr}")
        return Response(content="Inference failed", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Parse the transcription from the output
    transcription = parse_transcription(result.stdout)

    # Log the transcription
    logger.info(f"Transcription: {transcription}")

    return HTTPStatus.OK


def parse_transcription(output):
    """
    Parses the transcription from the stdout output of run.py
    """
    # Assuming the output contains lines like "prediction: <transcription>"
    lines = output.strip().split('\n')
    for line in lines:
        if line.startswith('prediction:'):
            transcription = line.replace('prediction:', '').strip()
            return transcription
    # If transcription is not found, return entire output
    return output

if __name__ == "__main__":
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", 6789))
    uvicorn.run(server, host=host, port=port)
