from moviepy.editor import AudioFileClip
import os
import random
import wave
import numpy as np
import struct
import matplotlib.pyplot as plt

datadir = "D:\\enterface database" # 你的数据集路径
sum = 1 # 记录视频文件数

# 定义函数 实现两通道wav转一通道
def twotoone(file_dir):
    # 只读方式打开WAV文件
    wf = wave.open(file_dir, 'rb')

    nframes = wf.getnframes()
    framerate = wf.getframerate()
    str_data = wf.readframes(nframes)
    sample_width = wf.getsampwidth()
    wf.close()

    # 将波形数据转换成数组
    wave_data = np.fromstring(str_data, dtype=np.short)
    print(wave_data.shape)
    wave_data.shape = (-1, 2)
    wave_data = wave_data.T
    mono_wave = (wave_data[0]+wave_data[1])/2
    print(mono_wave)

    time = np.arange(0, nframes)*(1.0/framerate)

    # save wav file
    wf_mono = wave.open(file_dir,'wb')
    wf_mono.setnchannels(1)
    wf_mono.setframerate(framerate)
    wf_mono.setsampwidth(sample_width)
    for i in mono_wave:
        data = struct.pack('<h', int(i))
        wf_mono.writeframesraw( data )
    wf_mono.close()

# 从数据集路径里找到所有的avi文件
for root,dirs,files in os.walk(datadir):
    # for dir in dirs:
    #     print(os.path.join(root,dir))
    for file in files:
        file_path = os.path.join(root, file)
        if file_path[-3:] == "avi" :

            sum = sum+1 #视频数量+1
            print(file_path)
            save_path = file_path[:-3] + "wav"
            print(save_path)
            # 转换成wav.(two channel) 保存到原路径文件夹
            my_audio_clip = AudioFileClip(file_path)
            my_audio_clip.write_audiofile(save_path)
            # 调用函数 将save_path 的 wav再转换成一通道
            twotoone(save_path)

print(sum)#输出遍历的视频总数

