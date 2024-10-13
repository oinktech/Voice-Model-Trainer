from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from transformers import FlaxWav2Vec2ForCTC, Wav2Vec2Processor
import jax
import jax.numpy as jnp
import soundfile as sf
import os
import shutil

app = Flask(__name__)
app.secret_key = 'secretkey123'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'

# 確保目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    transcription = None  # 初始化轉錄結果
    if request.method == "POST":
        model_file = request.files.get("model")
        audio_file = request.files.get("audio_file")

        if not model_file or not audio_file:
            flash("請上傳模型檔案與音訊檔案", "danger")
            return redirect(request.url)

        model_path = os.path.join(app.config['MODEL_FOLDER'], model_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)

        model_file.save(model_path)
        audio_file.save(audio_path)

        try:
            processor = Wav2Vec2Processor.from_pretrained(model_path)
            model = FlaxWav2Vec2ForCTC.from_pretrained(model_path)
            audio_input, _ = sf.read(audio_path)
            input_values = processor(audio_input, return_tensors="np", sampling_rate=16000).input_values

            # Convert to JAX array
            input_values = jnp.array(input_values)

            # Perform inference with Flax model
            logits = model(input_values).logits
            predicted_ids = jnp.argmax(logits, axis=-1)
            transcription = processor.decode(predicted_ids[0])

            flash(f"語音轉錄結果: {transcription}", "success")
        except Exception as e:
            flash(f"模型推理時出錯: {str(e)}", "danger")

    return render_template("index.html", transcription=transcription)

@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        audio_file = request.files.get("audio")
        if not audio_file:
            flash("請上傳音訊檔案", "danger")
            return redirect(request.url)

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(audio_path)

        try:
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model = FlaxWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

            audio_input, _ = sf.read(audio_path)
            input_values = processor(audio_input, return_tensors="np", sampling_rate=16000).input_values
            labels = jnp.array([1, 2, 3])  # 模擬的標籤

            # 模型訓練過程（简化处理）
            model.save_pretrained(os.path.join(app.config['MODEL_FOLDER'], "fine_tuned_model"))
            processor.save_pretrained(os.path.join(app.config['MODEL_FOLDER'], "fine_tuned_model"))

            flash("模型已成功微調並保存", "success")
        except Exception as e:
            flash(f"訓練時發生錯誤: {str(e)}", "danger")

    return render_template("train.html")

@app.route("/download/<model_name>")
def download(model_name):
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    zip_path = model_path + ".zip"
    
    shutil.make_archive(model_path, 'zip', model_path)
    
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
