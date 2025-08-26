import random
import numpy as np
import torch
import gradio as gr
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model

def chunk_text(text, max_chars=300):
    """Split long text into smaller chunks without breaking words."""
    words = text.split()
    chunks, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    # 🔹 Split long text into chunks
    text_chunks = chunk_text(text, max_chars=300)

    all_audio = []
    for chunk in text_chunks:
        wav = model.generate(
            chunk,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfgw,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        all_audio.append(wav.squeeze(0))

    # 🔹 Concatenate all chunks
    full_audio = torch.cat(all_audio, dim=-1)

    return (model.sr, full_audio.numpy())

with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Paste unlimited text here...",
                label="Text to synthesize",
                lines=12,   # larger text area
                placeholder="Enter your text (no character limit)"
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
