import os
import sys
import gradio as gr
from inference import Model

def get_available_models():
    checkpoints_dir = "checkpoints"
    local_models = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
    remote_models = ["metncelik/tr-lyrics-generator-cosmos-gpt2-large", "metncelik/tr-lyrics-generator-gpt2-uncased"]
    return local_models + remote_models

def generate_lyrics(model_name, prompt, max_length, temperature, top_k, top_p):
    model_path = model_name.split("/")[-1] if model_name.startswith("checkpoints") else model_name
    model = Model(model_path)
    
    lyrics = model.generate_lyrics(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    return lyrics

# Create the Gradio interface
with gr.Blocks(title="Şarkı Sözü Oluşturucu") as demo:
    gr.Markdown("# 🎵 Şarkı Sözü Oluşturucu")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                label="Model Seçiniz",
                value=get_available_models()[0] if get_available_models() else None
            )
            
            prompt_input = gr.Textbox(
                label="Şarkı sözü giriniz",
                placeholder="Şarkı sözlerini buraya yazınız veya boş bırakınız",
                lines=5
            )
            
            with gr.Accordion("Gelişmiş Ayarlar", open=False):
                max_length = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=150,
                    step=10,
                    label="Maksimum Uzunluk"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="Sıcaklık"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=75,
                    step=1,
                    label="Top K"
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top P"
                )
            
            generate_btn = gr.Button("Oluştur", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Oluşturulan Şarkı Sözleri",
                lines=15,
                show_copy_button=True
            )
    
    generate_btn.click(
        fn=generate_lyrics,
        inputs=[
            model_dropdown,
            prompt_input,
            max_length,
            temperature,
            top_k,
            top_p
        ],
        outputs=output
    )

if __name__ == "__main__":
    share = len(sys.argv) > 1 and ("--share" in sys.argv)
    demo.launch(share=share)
