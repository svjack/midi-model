# Midi-Model

## Midi event transformer for music generation

![](./banner.png)

# MIDI Composer Demo

This repository contains a demo for a MIDI composer model. The setup instructions below will guide you through the process of setting up the environment and running the demo.

## Setup Environment

1. **Clone the Repository**:
   First, clone the repository from Hugging Face Spaces:

   ```bash
   git clone https://huggingface.co/spaces/skytnt/midi-composer
   ```

2. **Navigate to the Repository Directory**:
   Change to the cloned repository directory:

   ```bash
   cd midi-composer
   ```

3. **Install Required Packages**:
   Install the necessary system packages and Python dependencies:

   ```bash
   sudo apt-get update && sudo apt-get install fluidsynth
   pip install -r requirements.txt
   pip install spaces
   ```

## Run the Demo

Once the environment is set up, you can run the demo using the following command:

```bash
python app.py --max-gen 4096 --share
```

This command will start the application with a maximum generation limit of 4096 and share the application publicly.


## Call API 
```python
from gradio_client import Client, handle_file

def generate_and_render_audio(model_name, instruments, drum_kit, input_midi_path=None, bpm=0, time_sig="auto", key_sig="auto", midi_events=128, gen_events=2048, temp=1.0, top_p=0.95, top_k=20, should_render_audio=True):
    """
    生成 MIDI 并渲染音频的函数。

    参数:
        model_name (str): 模型名称。
        instruments (list): 乐器列表。
        drum_kit (str): 鼓组名称。
        input_midi_path (str): 输入 MIDI 文件路径（可选）。
        bpm (int): 每分钟节拍数（0 表示自动）。
        time_sig (str): 时间签名（如 "4/4"）。
        key_sig (str): 调号（如 "C"）。
        midi_events (int): 使用前 n 个 MIDI 事件作为提示。
        gen_events (int): 生成的最大 MIDI 事件数。
        temp (float): 温度参数。
        top_p (float): top-p 采样参数。
        top_k (int): top-k 采样参数。
        should_render_audio (bool): 是否渲染音频。

    返回:
        tuple: 包含生成的音频文件路径的元组。
    """
    # 初始化客户端
    client = Client("http://127.0.0.1:7860")
    #client = Client("https://36a57c505482588881.gradio.live")
    
    # 调用 /run API 生成 MIDI
    run_result = client.predict(
        model_name=model_name,
        continuation_select="all",
        instruments=instruments,
        drum_kit=drum_kit,
        bpm=bpm,
        time_sig=time_sig,
        key_sig=key_sig,
        mid=handle_file(input_midi_path) if input_midi_path else None,
        midi_events=midi_events,
        reduce_cc_st=True,
        remap_track_channel=True,
        add_default_instr=True,
        remove_empty_channels=False,
        seed=0,
        seed_rand=True,
        gen_events=gen_events,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        allow_cc=True,
        api_name="/run"
    )

    # 提取生成的 MIDI 序列
    mid_seq = run_result[1]

    # 调用 /render_audio API 渲染音频
    render_result = client.predict(
        model_name=model_name,
        should_render_audio=should_render_audio,
        api_name="/render_audio"
    )

    # 返回渲染的音频文件路径

    
    return render_result

# 示例参数
model_name = "generic pretrain model (tv2o-medium) by skytnt"
instruments = ["Acoustic Grand", "Electric Piano 1"]
drum_kit = "Standard"
#input_midi_path = "path/to/your/input.mid"  # 可选
bpm = 120
time_sig = "4/4"
key_sig = "C"
midi_events = 128
gen_events = 2048
temp = 1.0
top_p = 0.95
top_k = 20
should_render_audio = True

# 调用函数生成并渲染音频
audio_files = generate_and_render_audio(
    model_name=model_name,
    instruments=instruments,
    drum_kit=drum_kit,
    #input_midi_path=input_midi_path,
    bpm=bpm,
    time_sig=time_sig,
    key_sig=key_sig,
    midi_events=midi_events,
    gen_events=gen_events,
    temp=temp,
    top_p=top_p,
    top_k=top_k,
    should_render_audio=should_render_audio
)

# 输出音频文件路径
print("Generated audio files:", audio_files)

import os
import shutil

def copy_and_rename_audio_files(audio_files, target_folder, prefix="output"):
    """
    将音频文件从临时文件夹拷贝到目标文件夹并重命名。

    参数:
        audio_files (tuple): 包含音频文件路径的元组。
        target_folder (str): 目标文件夹路径。
        prefix (str): 重命名文件的前缀，默认为 "output"。

    返回:
        list: 包含新文件路径的列表。
    """
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 存储新文件路径
    new_file_paths = []

    # 遍历音频文件
    for i, audio_file in enumerate(audio_files):
        if audio_file:  # 确保文件路径不为空
            # 生成新文件名
            new_file_name = f"{prefix}_{i + 1}.mp3"
            new_file_path = os.path.join(target_folder, new_file_name)

            # 拷贝文件
            shutil.copy(audio_file, new_file_path)
            new_file_paths.append(new_file_path)

    return new_file_paths

copy_and_rename_audio_files(audio_files, target_folder = "demo0", prefix="output")
```

- MIDI File Render
- https://huggingface.co/spaces/asigalov61/Advanced-MIDI-Renderer


## Updates
- v1.3: MIDITokenizerV2 and new MidiVisualizer
- v1.2 : Optimise the tokenizer and dataset. The dataset was filtered by MIDITokenizer.check_quality. Using the higher quality dataset to train the model, the performance of the model is significantly improved.

## Demo

- [online: huggingface](https://huggingface.co/spaces/skytnt/midi-composer)

- [online: colab](https://colab.research.google.com/github/SkyTNT/midi-model/blob/main/demo.ipynb)

- [download windows app](https://github.com/SkyTNT/midi-model/releases)

## Pretrained model

[huggingface](https://huggingface.co/skytnt/midi-model-tv2o-medium)

## Dataset

[projectlosangeles/Los-Angeles-MIDI-Dataset](https://huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset)

## Requirements

- install [pytorch](https://pytorch.org/)(recommend pytorch>=2.0)
- install [fluidsynth](https://www.fluidsynth.org/)>=2.0.0
- `pip install -r requirements.txt`

## Run app

`python app.py`

## Train 

`python train.py`
 
## Citation

```bibtex
@misc{skytnt2024midimodel,
  author = {SkyTNT},
  title = {Midi Model: Midi event transformer for symbolic music generation},
  year = {2024},
  howpublished = {\url{https://github.com/SkyTNT/midi-model}},
}
