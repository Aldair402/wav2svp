import gradio as gr
from infer import wav2svp

def inference(input, bpm, extract_pitch, extract_tension, extract_breathiness):
    model_path = "weights/model_steps_64000_simplified.ckpt"
    return wav2svp(input, model_path, bpm, extract_pitch, extract_tension, extract_breathiness)

def webui():
    with gr.Blocks() as webui:
        gr.Markdown('''<div align="center"><font size=6><b>wav2svp - Waveform to Synthesizer V Project</b></font></div>''')
        gr.Markdown("Upload an audio file and download the svp file with midi and selected datas.")
        with gr.Row():
            with gr.Column():
                input = gr.File(label="Input Audio File", type="filepath")
                bpm = gr.Number(label='BPM Value', minimum=20, maximum=200, value=120, step=0.01, interactive=True)
                extract_pitch = gr.Checkbox(label="Extract Pitch Data", value=True)
                extract_tension = gr.Checkbox(label="Extract Tension Data (Experimental)", value=False)
                extract_breathiness = gr.Checkbox(label="Extract Breathiness Data (Experimental)", value=False)
                run = gr.Button(value="Generate svp File", variant="primary")
            with gr.Column():
                output_svp = gr.File(label="Output svp File", type="filepath", interactive=False)
                output_midi = gr.File(label="Output midi File", type="filepath", interactive=False)
        run.click(inference, [input, bpm, extract_pitch, extract_tension, extract_breathiness], [output_svp, output_midi])
    webui.launch(inbrowser=True, share=True)

if __name__ == '__main__':
    webui()
