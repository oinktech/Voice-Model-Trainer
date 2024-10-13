from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
import torch
import os
import soundfile as sf
from datasets import Dataset
from celery import Celery
import shutil

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 16 MB file size limit

# 确保上传和模型目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Celery 配置
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

@celery.task
def train_model(audio_path):
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        audio_input, _ = sf.read(audio_path)
        input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values
        labels = torch.tensor([1, 2, 3])  # 模拟标签，替换为实际数据

        train_dataset = Dataset.from_dict({"input_values": [input_values], "labels": [labels]})

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            num_train_epochs=1,
            logging_dir="./logs",
            logging_steps=10
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

        model_path = os.path.join(app.config['MODEL_FOLDER'], "fine_tuned_model")
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)
        return f"Model fine-tuned and saved successfully at {model_path}"
    except Exception as e:
        return f"Error during training: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_file = request.files.get("model")
        audio_file = request.files.get("audio_file")

        if not model_file or not audio_file:
            flash("Please upload both a model and an audio file.", "danger")
            return redirect(request.url)

        model_path = os.path.join(app.config['MODEL_FOLDER'], model_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)

        model_file.save(model_path)
        audio_file.save(audio_path)

        try:
            processor = Wav2Vec2Processor.from_pretrained(model_path)
            model = Wav2Vec2ForCTC.from_pretrained(model_path)
            audio_input, _ = sf.read(audio_path)
            input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])

            flash(f"Transcription: {transcription}", "success")
        except Exception as e:
            flash(f"Error during model inference: {str(e)}", "danger")

    return render_template("index.html")

@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        audio_file = request.files.get("audio")
        if not audio_file:
            flash("Please upload an audio file.", "danger")
            return redirect(request.url)

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(audio_path)

        # 启动后台训练任务
        task = train_model.apply_async(args=[audio_path])
        flash("Training started, you will be notified once it's done.", "success")

    return render_template("train.html")

@app.route("/download/<model_name>")
def download(model_name):
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    zip_path = model_path + ".zip"
    
    shutil.make_archive(model_path, 'zip', model_path)
    
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=10000)
