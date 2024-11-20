<div style="text-align: center;">

# wav2svp: Waveform to Synthesizer V Project

</div>

### Description

wav2svp is a project that converts a waveform to a Synthesizer V Project (SVP) file. It is based on the [SOME](https://github.com/openvpi/SOME) and [RMVPE](https://github.com/Dream-High/RMVPE). In addition to automatically extracting MIDI, this project can also extract **pitch lines** simultaneously. But unfortunately, at present, it's unable to simultaneously extract lyrics.

### Usage

You can download the **One click startup package** from [releases](https://github.com/SUC-DriverOld/wav2svp/releases), unzip and double click `go-webui.bat` to start the WebUI.

### Run from Code

1. Clone this repository and install the dependencies. We recommand to use python 3.10.

    ```shell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

3. Download pre-trained models:

    - [0119_continuous128_5spk](https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip) and unzip it to `weights`.
    - [rmvpe](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) and unzip it to `weights` and rename it to `rmvpe.pt`.
    - Order the `weights` folder as follows:

    ```shell
    weights
      ├-config.yaml
      ├-model_steps_64000_simplified.ckpt
      └-rmvpe.pt
    ```

4. Run the following command to start WebUI:

    ```shell
    python webui.py
    ```

5. You can download the inference results from WebUI interface or from the `results` folder.

### Command Line Usage

Use `infer.py`:

```shell
usage: infer.py [-h] [--tempo TEMPO] audio_path model_path

Inference for wav2svp

positional arguments:
  audio_path     Path to the input audio file

options:
  -h, --help     show this help message and exit
  --model_path     Path to the model file, default: weights/model_steps_64000_simplified.ckpt
  --tempo TEMPO  Tempo value for the midi file, default: 120
```

You can find the results in the `results` folder.

### Thanks

- [openvpi/SOME] [openvpi/SOME](https://github.com/openvpi/SOME)
- [Dream-High/RMVPE] [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE)
- [yxlllc/RMVPE] [yxlllc/RMVPE](https://github.com/yxlllc/RMVPE)