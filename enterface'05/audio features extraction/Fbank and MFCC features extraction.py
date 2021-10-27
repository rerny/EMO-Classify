import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.fftpack import dct
from scipy.io import wavfile
import python_speech_features

def extraction(audio_file):
    sample_rate, signal = wavfile.read(audio_file)
    # sample_rate:WAV文件的采样率 signal:从wav文件读取的数据
    print('sample_rate:{}, len:{}'.format(sample_rate, len(signal)))
    # 取前2s
    signal = signal[: int(2 * sample_rate)]
# 预加重(Pre-Emphasis)  y(t) = x(t) - a * x(t - 1)   为了平衡高低频数据
    pre_emphasis = 0.97  # usually 0.95 or 0.97
    emphasized_signal = np.append(
            signal[0],
            signal[1:] - pre_emphasis * signal[: -1]
        ) #第一维度开始到最后一维度 - 第0维度到倒数第二维度的数据*0.97
    # 利用pylab画出音频振幅
    n_frames = len(emphasized_signal)
    time = np.arange(0, n_frames) * (1.0 / sample_rate)
    pl.subplot(1, 2, 1)
    pl.plot(time, signal)
    pl.xlabel('time (seconds)')
    pl.ylabel('amplitude') #振幅
    pl.title('init audio')
    pl.subplot(1, 2, 2)
    pl.plot(time, emphasized_signal)
    pl.xlabel('time (seconds)')
    pl.ylabel('amplitude')
    pl.title('emp audio')
    pl.show()
# 分帧(Framing)
    #     here, params set as follows:
    #     窗口长度 frame_size = 0.025(s), it menas 8kHz signal has 0.025 * 8000 = 200 samples.
    #     时间间隔 frame_stride = 0.01(s), 0.01 * 8000 = 80 samples.
    #     相邻两帧之间有一部分重叠 overlap = 0.015(s), 0.015 * 8000 = 125 samples.
    frame_size, frame_stride, overlap = 0.025, 0.01, 0.015  # Convert from seconds to samples
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    frame_length, frame_step = int(round(frame_length)),int(round(frame_step)) #小数位数进行四舍五入
    #frame_length 每个frame的采样率          frame_step 相邻间隔的采样率
    print("frame_length = ",frame_length)
    print("frame_step = ",frame_step)
    singal_length = len(emphasized_signal) #获得音频数据的长度  28000
    print("填充前的singal长度：",singal_length)
    num_frames = int(np.ceil(  #np.ceil 返回数字的上入整数
        float(np.abs(singal_length - frame_length)) / frame_step))  # 计算出总共取了多少个 frame >=1
    #用0矩阵填充singal 使得整个singal 能完整地被采样
    pad_singal_length = (num_frames) * frame_step + frame_length  #    28000->28040
    z = np.zeros((pad_singal_length - singal_length))
    print("填充完毕->")
    print("填充后的singal长度：",pad_singal_length)
    pad_singal = np.append(emphasized_signal, z)
    #
    # 音频信号转化为二维矩阵 每一行即是一个音频帧的内容
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step),(frame_length, 1) ).T
    frames = pad_singal[indices.astype(np.int32, copy=False)]
    print("分帧完毕->")
    print("每个frame的shape：",frames.shape)
# 加窗(Window) 对每个帧 应用 汉明窗窗口函数
    print("hamming window 's frame_length (x) = ",frame_length)
    window = np.hamming(frame_length) #加权的余弦函数 具有唯一参数 横坐标输出点的个数
    plt.plot(window)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title("Hamming window")
    plt.show()
    frames *= window
    print("加窗结束->")
    print("---音频预处理结束---")
# 傅立叶变换FFT->梅尔滤波组
    # P = |FFT(Xi)|^2 / N, Xi is ith frame of signal x
    NFFT = 512
    mag_frames = np.absolute((np.fft.rfft(frames, NFFT)))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    print("FFT变化得到能量谱结束->")
# Mel滤波器(三角滤波器) 再取log 得到fbank
    # 26 filters, nfilt = 26 on a Mel-scale to the power spectrum to extract frequency bands.
    # convert frequency(f) and Mel(m) with equations:
    # m = 2595 * log10(1 + f / 700)
    # f = 700 * (10 ^ (m / 2595) - 1)
    n_filters = 26 # 26个滤波器来提取26维的fbank
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  # need 26 filters banks, so need 28 points
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # convert Mel to Hz
    # bin = sample_rate / NFFT  # fequency bin equation
    bins = np.floor((NFFT + 1) * hz_points / sample_rate)  # hz_points / bin

    fbank = np.zeros((n_filters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus = int(bins[m - 1])
        f_m = int(bins[m])
        f_m_plus = int(bins[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

#对数运算
    filter_banks = np.dot(pow_frames, fbank.T) #功率谱与滤波器做点积 降维
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks) #将filter_bank中的0值改为最小负数，防止运算出现问题
    filter_banks = 20 * np.log10(filter_banks) #log梅尔频谱

    print("filter_banks.shape = ",filter_banks.shape)
    pl.plot(filter_banks)
    pl.subplot(1, 1, 1)
    pl.xlabel("Frequency")
    pl.ylabel("Time")
    pl.title("Spectrogram")
    pl.show()
    print("提取26维fbanks结束->")

# 提取13MFCC特征 即FBank特征的基础上再进行离散余弦变换DCT
    num_ceps = 13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # keep 2-14 13维度
    #apply sinusoidal liftering to the MFCCs to de-emphasize higher MFCCs
    (n_frames, n_coeff) = mfcc.shape
    n = np.arange(n_coeff)
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    print(mfcc.shape)
    pl.plot(mfcc)
    pl.xlabel("Time")
    pl.ylabel("MFCC Coefficients")
    pl.title("MFCCs")
    pl.show()
    # plt.title("mfcc")
    # plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.05, extent=[0,mfcc.shape[1],0,mfcc.shape[0]]) #画热力图
    # plt.xlabel("Frames",fontsize = 14)
    # plt.ylabel("Dimension",fontsize = 14)
    # plt.tick_params(axis='both',labelsize = 14)
    # plt.show()
    print("提取13维MFCC结束->")
#MFCC 1,2阶级差分
    d_mfcc_feat = python_speech_features.delta(mfcc, 1)
    d_mfcc_feat2 = python_speech_features.delta(mfcc, 2)
    print(d_mfcc_feat.shape)
    print(d_mfcc_feat2.shape)
#拼接MFCC 标准+1阶+2阶
    feature = np.hstack((mfcc, d_mfcc_feat, d_mfcc_feat2))
    print("feature.shape:",feature.shape)

audio_file = "D:/enterface database/subject 1/anger/sentence 1/s1_an_1.wav"
extraction(audio_file)
print("get features successfully")





