from flask import Flask, render_template, request, jsonify
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

app = Flask(__name__)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    genre = request.form.get('genre')
    tempo = request.form.get('tempo')

    prompt = f"{genre} music at {tempo} BPM"
    
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    audio_values = model.generate(**inputs, max_new_tokens=256)

    output_path = "generated_music.wav"
    wavfile.write(output_path, rate=model.config.audio_encoder.sampling_rate, data=audio_values[0, 0].numpy())

    return jsonify({'generated_music': output_path})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
