# Sys libs
import tensorflow as tf
from morph_net.network_regularizers import flop_regularizer
from morph_net.network_regularizers import latency_regularizer
from morph_net.tools import structure_exporter
import numpy as np
from datetime import datetime
import time
import cv2
import copy
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import argparse
import logging
import utils2
from sfd.sfd_detector import SFDDetector
from meta_dataset import Meta_Dataset
from tracker import tracker_detector
from scipy.interpolate import interp1d

def conv2d(input, w, b):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def conv2da(input, w, b):
  return tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(input):
  return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

def fc(input, w, b):
  return tf.nn.relu(tf.add(tf.matmul(input, w), b))

def NetFix(_input_r, _weights, _biases, _keep_prob):
  
  with tf.name_scope('conv1'):
    _conv1 = conv2d(_input_r, _weights['wc1'], _biases['bc1'])
  
  with tf.name_scope('pool1'):
    _pool1 = max_pool(_conv1)

  with tf.name_scope('conv2'):
      _conv2 = conv2d(_pool1, _weights['wc2'], _biases['bc2'])

  with tf.name_scope('conv3'):
      _conv3 = conv2d(_pool2, _weights['wc3'], _biases['bc3'])

  with tf.name_scope('conv4'):
      _conv4 = conv2da(_conv3, _weights['wc4'], _biases['bc4'])

  with tf.name_scope('conv5'):
      _conv5 = conv2da(_conv4, _weights['wc5'], _biases['bc5'])

  with tf.name_scope('gru1'):
	_gru = tf.keras.layers.GRU(4)
	_output = gru(_conv4)
	_gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
	_hid, final_state = _gru(_conv4)

  with tf.name_scope('conv6'):
	_conv5 = conv2da(_conv4, _weights['wc5'], _biases['bc5'])
	ZeroPadding2D (( 0,1 ,0, 1 ))
	#_densel = tf.reshape(_pool3,[-1, _weights['wd1'].get_shape().as_list()[0]])
	_interp = tfp.math.interp_regular_1d_grid(_conv5,32,32)
	
#with tf.name_scope('pool2'):
     #_pool3 = max_pool(_conv5)

  #with tf.name_scope('fc1'):
      #_fc1 = fc(_densel, _weights['wd1'], _biases['bd1'])
      #_fc1_drop = tf.nn.dropout(_fc1, _keep_prob)

  #with tf.name_scope('fc2'):
      #_fc2 = fc(_fc1_drop, _weights['wd2'], _biases['bd2'])
     # _fc2_drop = tf.nn.dropout(_fc2, _keep_prob)

  #with tf.name_scope('out'):
      #_out = tf.add(tf.matmul(_fc2_drop, _weights['wd3']), _biases['bd3'])

  return _out
  

 #-----------simple test-------------#
pred = NetFix(x, weights, biases, keep_prob)
n_output = 10
learning_rate = 0.001
dropout = 0.75
train_data, test_data = tf.keras.datasets.mnist.load_data()
X_train, X_test = train_data[0], test_data[0]
Y_train, Y_test = train_data[1], test_data[1]
train_dir = "./Morph_save/"
print("CNN READY")

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
print(pred.op)
#--------------------------------------#
 

#initial model parameters  
weights = {
'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 32], dtype=tf.float32, stddev=0.1), name='weights1'),
'wc2': tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=0.1), name='weights2'),
'wc3': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=0.1), name='weights3'),
'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=0.1), name='weights4'),
'wc5': tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=0.1), name='weights5'),
'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=0.1), name='weights4'),
'wc5': tf.Variable(tf.truncated_normal([3, 3, 64, 16], dtype=tf.float32, stddev=0.1), name='weights5'),
'wd1': tf.Variable(tf.truncated_normal([4*4*128, 1024], dtype=tf.float32, stddev=0.1), name='weights_fc1'),
'wd2': tf.Variable(tf.random_normal([1024, 1024], dtype=tf.float32, stddev=0.1), name='weights_fc2'),
'wd3': tf.Variable(tf.random_normal([1024, n_output], dtype=tf.float32, stddev=0.1), name='weights_output')
}
  
  
biases = {
'bc1': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases1'),
'bc2': tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases2'),
'bc3': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases3'),
'bc4': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases4'),
'bc5': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases5'),
'bd1': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases_fc1'),
'bd2': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases_fc2'),
'bd3': tf.Variable(tf.constant(0.0, shape=[n_output], dtype=tf.float32), trainable=True, name='biases_output')
}  
	
print("CNN READY")

#----------Morp-net-------------#
network_regularizer = flop_regularizer.GammaFlopsRegularizer(
     output_boundary = [pred.op],
     input_boundary = [x.op, y.op],
     gamma_threshold = 1e-3
)

regularization_strength = 1e-8
regularizer_loss = (network_regularizer.get_regularization_term() * regularization_strength)

print(y)

print(pred)


#test_lost
model_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y,1),logits=pred))

cost_op = network_regularizer.get_cost()
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
train_op = optimizer.minimize(model_loss + regularizer_loss)


exporter = structure_exporter.StructureExporter(network_regularizer.op_regularizer_manager)
init = tf.global_variables_initializer()
print("FUNCTIONS READY")
	
	
	
	
	
class Kalman(object):
    def __init__(self, x, y, fs,):
        self.filt = KalmanFilter(dim_x=2, dim_z=2)
        self.filt.x = np.array([x, y])
        self.filt.H = np.array([[1., 0.], [0., 1.]])
        if 1 == 2:
            self.filt.P *= 1000.

    def process(self, x, y):
        z = np.array([x, y])
        z = np.reshape(z, (1, 2))
        self.filt.predict()
        rez = self.filt.x
        self.filt.update(z)
        return rez  # self.filt.x

def decrease_optimizer_rate(optimizer):
    for i in optimizer.param_groups:
        i['lr'] *= 0.998
    if i['lr'] < 1e-5 / 10:
        i['lr'] = 1e-5 / 10
    # print ('learning rate =', i['lr'])

def save_plt_image(fig, fname):
    import pickle
    pickle.dump(fig, open(fname, 'wb'))
    plt.close(fig)

def mean(data):
    dim = 1
    for i in range(data.dim()):
        dim *= data.size(i)
    return tf.add_n(data) / dim

def std_var(data):
    m = mean(data)
    var = mean(tf.pow(data, 2)) - tf.pow(m, 2)
    return (m, tf.sqrt(var))

def gen_freqs(median_freqs, data,bnum, f_low, f_high, f_step, df):
    centers = []
    dfs = []
    centers = median_freqs[bnum]  # [np.mean(data[bnum][2])*f_step]
    dfs = df / 2  # [10/60]
    # print ('centers;', centers)

    c_f = []
    all_f = []
    for i in range(1):
        f = np.arange(centers - dfs, centers + dfs, f_step)
        f2 = np.arange(f_low, centers - dfs, f_step)
        f22 = np.arange(centers + dfs, f_high, f_step)
        f2 = np.concatenate((f2, f22), axis=0)

        f = np.expand_dims(f, axis=0)
        f2 = np.expand_dims(f2, axis=0)

        c_f += [f]
        all_f += [f2]
    c_f = np.concatenate(c_f, axis=0)[0]
    all_f = np.concatenate(all_f, axis=0)[0]

    return (c_f, all_f)

def analyse_psd(freq_, psd_):
    freq = np.array(freq_)
    psd = np.array(psd_)
    psd = psd[:, 0]
    idx = np.where(freq > 0.6)[0][0]
    psd2 = psd[idx:]
    freq2 = freq[idx:]
    idx2 = np.argmax(psd2)
    return freq2[idx2]

class Trainer(object):
    def __init__(self, dataset, net, basename):
        self.basename = basename
        self.model = net
        self.dataset = dataset
        self.batch_size = None
        self.batch_seq = 10

        self.criterion = tf.nn.l2_loss()
        self.optimizer = optimizer
    def init_kalmans(self, rects, fs):
        self.kalmans_c = []
        self.kalmans_s = []
        for i in range(len(rects)):
            rect = rects[i][0]
            wx = rect[2] - rect[0]
            wy = rect[3] - rect[1]
            cx = rect[0] + wx / 2
            cy = rect[1] + wy / 2
            self.kalmans_c += [Kalman(cx, cy, fs)]
            self.kalmans_s += [Kalman(wx, wy, fs)]
        self.start_size = [wx, wy]

    def update_kalmans(self, rects):
        rez = []
        for i in range(len(rects)):
            rect = rects[i][0]
            wx = rect[2] - rect[0]
            wy = rect[3] - rect[1]
            cx = rect[0] + wx / 2
            cy = rect[1] + wy / 2
            (nwx, nwy) = self.kalmans_s[i].process(wx, wy)
            (ncx, ncy) = self.kalmans_c[i].process(cx, cy)

            nwx = self.start_size[0]
            nwy = self.start_size[1]

            rez += [[[ncx - nwx / 2, ncy - nwy / 2, ncx + nwx / 2, ncy
                    + nwy / 2]]]
        return rez

    def normalize_video(self, data):
        mean = np.zeros([self.batch_size, data[0][0].shape[1],
                        data[0][0].shape[2], data[0][0].shape[3]])
        msq = copy.copy(mean)
        for i in range(len(data)):
            d = np.concatenate(data[i], axis=0)
            d = d.astype('float')
            mean += d
            msq += np.power(d, 2)
        m = mean / len(data)
        sig = np.sqrt(msq / len(data) - np.power(m, 2))
        return (m, sig)

    def estimate_hr(self, data, fs):
        from scipy import signal
        (f, Pxx_spec) = signal.welch(data, fs, window='hanning',
                nperseg=1024, scaling='spectrum')
        idx = np.where(f > 39 / 60)
        idx = idx[0][0]
        pos = np.argmax(Pxx_spec[idx:]) + idx
        freq = f[pos]
        return freq

    def mk_target_hr_ppg(self, data, fs):
        out = []
        for i in range(self.batch_size):
            ppg1 = np.array(data[i][1])
            ppg1 = self.estimate_hr(ppg1, fs)
            out += [float(ppg1)]
        return out

    def mk_target_time(self, data):
        out = []
        for i in range(self.batch_size):
            ppg1 = data[i][0]
            # print ('ppg1:', ppg1.shape)
            out += [ppg1]
        out = np.array(out)
        # print ('out:', out.shape)
        return out

    def mk_morph_serie(self, t_min, t_max, length):
        n = int(np.random.uniform(3, 25))
        basev = (t_max - t_min) / 2 + t_min
        base = np.zeros(length) + basev
        pos = [0]
        for i in range(n):
            if i > 0:
                if pos[-1] + 30 > length:
                    continue

            p = int(np.random.uniform(pos[-1] + 30, length - 1))
            pos += [p]

        pos[-1] = length - 1
        pos = list(dict.fromkeys(pos))

        for i in range(len(pos)):
            base[pos[i]] = np.random.uniform(t_min, t_max)

        for i in range(len(pos) - 1):
            steps = pos[i + 1] - pos[i]
            start = base[pos[i]]
            end = base[pos[i + 1]]
            dx = (end - start) / steps
            for j in range(steps):
                base[pos[i] + j] = start + dx * j
        return base

    def morph_video(self, video, theta_min=-5, theta_max=5, z_min=0.95, z_max=1.05, each_flag=0):
        for i in range(self.batch_size):
            if each_flag == 0:
                matrix = utils2.gen_random_morph(video[0][0],
                        theta_min=theta_min, theta_max=theta_max,
                        z_min=z_min, z_max=z_max)

            theta_seq = self.mk_morph_serie(theta_min, theta_max,
                    self.batch_seq)
            z_seq = self.mk_morph_serie(z_min, z_max, self.batch_seq)

            serie_flag = 0
            if np.random.uniform() > 0.7:
                serie_flag = 1
            serie_flag = 1
            # print ('morphing,serie flag:', serie_flag, 'video:',
            #        len(video))

            brightness_coef = np.random.uniform(-0.01, 0.01)
            brightness_x1x2 = [int(np.random.uniform(0, 20))]
            brightness_x1x2 += \
                [int(np.random.uniform(brightness_x1x2[-1]
                 + np.random.uniform(30, 40), len(video) - 40))]
            brightness_x1x2 += \
                [int(np.random.uniform(brightness_x1x2[-1]
                 + np.random.uniform(30, 40), len(video) - 1))]

            if np.random.uniform() > 0.3:
                brightness_coef = 0
            brightness_coef = 0
            # print ('brightness:', brightness_coef)

            for j in range(len(video)):
                if each_flag == 1:
                    if 1 == 1 and serie_flag == 0:
                        if np.random.uniform() > 0.7:
                            matrix = \
                                utils2.gen_random_morph(video[0][0],
                                    theta_min=theta_min,
                                    theta_max=theta_max, z_min=z_min,
                                    z_max=z_max)
                        else:
                            matrix = \
                                utils2.gen_random_morph(video[0][0],
                                    theta_min=-0.5, theta_max=0.5,
                                    z_min=0.98, z_max=1.01)
                    if 1 == 1 and serie_flag == 1:
                        matrix = utils2.gen_random_morph(video[0][0],
                                theta_min=theta_seq[j],
                                theta_max=theta_seq[j] + 0.01,
                                z_min=z_seq[j], z_max=z_seq[j] + 0.001)

                video[j][i][0] = \
                    utils2.random_morph_img(video[j][i][0], matrix)
                if j >= brightness_x1x2[0] and j <= brightness_x1x2[1]:
                    brightness_addon = brightness_coef \
                        / (brightness_x1x2[1] - brightness_x1x2[0]) \
                        * (j - brightness_x1x2[0])
                    video[j][i][0] = video[j][i][0] * (1
                            + brightness_addon)

                    bright_old = brightness_addon
                if j >= brightness_x1x2[1] and j <= brightness_x1x2[2]:
                    brightness_addon = -brightness_coef \
                        / (brightness_x1x2[2] - brightness_x1x2[1]) \
                        * (j - brightness_x1x2[1])
                    brightness_addon += bright_old
                    video[j][i][0] = video[j][i][0] * (1
                            + brightness_addon)
        return video

    def bw_video2(self, data):
        ret = []
        for i in range(len(data)):
            tmp = cv2.cvtColor(data[i][0], cv2.COLOR_RGB2GRAY)
            tmp = np.expand_dims(tmp, 2)
            tmp = np.expand_dims(tmp, 0)
            ret += [tmp]
        return ret

    def bw_video(self, batch_video):
        for i in range(len(batch_video)):
            for j in range(len(batch_video[i])):
                tmp = cv2.cvtColor(batch_video[i][j][0],
                                   cv2.COLOR_RGB2GRAY)
                batch_video[i][j] = \
                    np.zeros((batch_video[i][j].shape[0],
                             batch_video[i][j].shape[1],
                             batch_video[i][j].shape[2], 1))
                batch_video[i][j][0][:, :, 0] = tmp
        return batch_video
	
	
	n_output = 10
	learning_rate = 0.001
	dropout = 0.75
	train_data, test_data = tf.keras.datasets.mnist.load_data()
	X_train, X_test = train_data[0], test_data[0]
	Y_train, Y_test = train_data[1], test_data[1]
	train_dir = "./Morph_save/"
	print("CNN READY")

	x = tf.placeholder(tf.float32, [None, 28, 28, 1])
	y = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)
  

	def log10(x):
	  numerator = tf.log(x)
	  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	  return numerator / denominator

    def train_psd_class_r3(self):

        tot_loss = 0
        for epoch in range(epochs):
            print ('@@@@ epoch:', epoch)
            tot_loss = 0
            for iter in range(epoch_iter):
                print ('iter:', iter)
                (batch_video, batch_data, cur_fps) = \
                    self.dataset.get_batch_seq(self.batch_size,
                        self.batch_seq, self.dataset_step)
						
                batch_video = self.morph_video(
                    batch_video,
                    theta_min=-10,
                    theta_max=10,
                    z_min=0.95,
                    z_max=1.05,
                    each_flag=0,
                    )

                loss = 0

                out_seq = []

                self.model.hid = None
                self.model.est1.hid = None
                seq_len = self.model.seq_len

                all_heads = []

                out_extractor_seq = []

                bad_head = 0

                for seq in range(0, len(batch_video), 1):
                    utils2.progress_print(seq, len(batch_video))

                    np_batch = np.concatenate(batch_video[seq], axis=0)
                    pt_batch = \
               
                    pt_batch = pt_batch.permute(0, 3, 1, 2)
                    pt_batch = pt_batch.float()

                    (out_pt_batch, heads) = self.detect_heads(pt_batch)

                    for i in range(len(heads)):
                        if len(heads[i]) < 1:
                            bad_head = 1
                    if bad_head == 1:
                        break

                    boxes_r = None
                    boxes = heads

                    boxes_r = heads[0][0]
                    boxes_r = [boxes_r[0], boxes_r[1], boxes_r[2],
                               boxes_r[3]]

                    if seq == 0:
                        self.init_kalmans(heads, cur_fps[0])
                        beg_heads = copy.copy(heads)
                        tracker.init(batch_video[0][0][0][:, :, 0],
                                boxes_r)

                    if seq > 0:
                        heads2 = self.update_kalmans(heads)
                        heads = heads2

                    if boxes_r is not None:
                        tracker.update2(batch_video[0][0][0][:, :, 0],
                                boxes_r)
                    else:
                        tracker.update1(batch_video[0][0][0][:, :, 0])

                    heads = [[tracker.curr_pos]]

                    np_batch = \
                        np.concatenate(self.bw_video2(batch_video[seq]),
                            axis=0)
                    pt_batch = \
                      
                    pt_batch = pt_batch.permute(0, 3, 1, 2)
                    pt_batch = pt_batch.float()

                    pt_batch = pt_batch / 256
                    pt_batch = self.head_to_roi(pt_batch, pt_batch,
                            heads)

                    out = self.model(pt_batch)
                    out_extractor_seq += [out]

                out_seq = []

                counter = 0
                show1 = []
                if bad_head == 1:
                    continue

                for seq in range(self.model.seq_len,
                                 len(out_extractor_seq)
                                 - self.model.seq_len, 1):
                    utils2.progress_print(seq, len(batch_video))

                    D = tf.concat(out_extractor_seq[seq:seq + seq_len],
                                  dim=1)

                    D = D.view(D.size(0), -1)
                    out_net = self.model.est1(D)  
                    
                    if 1 == 1:
                        out_seq += [out_net]

                        show1 += [out_net[0][0].detach().cpu()]

                # seq loop ended
                if 1 == 1 and bad_head == 0:

                    f_high = cur_fps[0] / 2
                    f_step = f_high / self.batch_seq
                    # print ('fstep=', f_step)

                    f_step = 1 / 30  # 1/cur_fps/3

                    f_high = 3
                    f_low = 0.4

                    if 1 == 1:
                        typ = 1
                        median_freq = []
                        freq_dx = []

                        times = self.mk_target_time(batch_data)  # batch_data[:][0]

                        for i in range(self.batch_size):
                            hr_data2 = \
                                (batch_data[i][2])[self.model.seq_len:]
                            dt = np.array(times[i])
                            dt = dt[1:] - dt[:-1]
                            hr_data = hr_data2  # np.array(utils2.measure_freq(batch_data[i][1],1/np.mean(dt)))

                            median_freq += [np.median(hr_data) / 60]
                            freq_dx += [max(20 / 60, np.max(hr_data2)
                                    / 60 - np.min(hr_data2) / 60)]
									
                       
                            if num_b == 0:
							

                                fig = plt.figure()
                                s1 = freqs_all
                                s2 = psd_all.detach().cpu().numpy()
                                # print ('s1:', s1.shape, 's2:', s2.shape)
                                plt.plot(s1, s2, marker='^')

                                s1 = freqs_center
                                # print ('s1:', s1.shape, 's2:', s2.shape)
                                plt.plot(s1, s22, marker='>')

                                name = self.basename + '_' + str(typ) \
                                    + '_figure2.pickle'
                                # print ('saving image as:', name)
                                fig.suptitle(name, fontsize=16)
                                # save_plt_image(fig, name)

                            psd_all = tf.add_n(psd_all, dim=0)
                            
                            print ('psd_center:', psd_center)
                            print ('psd all:', psd_all)

                            if num_b == 0:
                                loss = -10 * log1010(psd_center
                                        / psd_all)
                            else:
                                loss = loss - 10 \
                                    * log1010(psd_center / psd_all)

                        # now classification loss
                        print ('loss:', loss.item())
                        loss.backward()

                        for b in range(self.batch_size):
                            print ('true heart rate:',
                                   batch_data[b][2][0])

                        self.optimizer.step()
                        self.model.zero_grad()
                        self.optimizer.zero_grad()

                        del batch_video
                        del out_seq
                        del pt_batch

                show2 = (batch_data[0][1])[self.psd_start:]
                # print ('show2:', show2.shape)

                show1 = np.array(show1[self.psd_start:])


                if iter % 2 == 0 and 1 == 1 and iter > -1:
                    # print ('heart rate:', batch_data[0][2][0])
                    fig = plt.figure()
                    s1 = show1
                    s2 = show2

                    plt.plot(s1, marker='v')
                    plt.plot(s2, marker='^')
                    name = self.basename + '_' + str(typ) \
                        + '_figure.pickle'
                    # print ('saving image as:', name)
                    fig.suptitle(name, fontsize=16)

                    # save_plt_image(fig, name)

                save_flag = 1
                if math.isnan(loss.item()) == True \
                    or math.isinf(loss.item()) == True:
                    save_flag = 0

                if iter % 10 == 0 and 1 == 1 and save_flag == 1:
                    decrease_optimizer_rate(self.optimizer)
					

                    tot_loss += loss.detach().item()

            nni.report_intermediate_result(tot_loss)

            for i in self.optimizer.param_groups:
                i['lr'] *= 0.998
                if i['lr'] < 1e-5 / 10:
                    i['lr'] = 1e-5 / 10
                print ('learning rate =', i['lr'])
            
        nni.report_final_result(tot_loss)
		
		
def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions
	
	
def prepare(args):
    global net
    global optimizer
    global epochs
    global epoch_iter
    
    epochs = 10
    epoch_iter = 20
    
    FClayers = []
    inputFeaturesCurrent = 1200
    if (args['MLPLayer0']['_name'] != "Empty"):
        FClayers.append(nn.Linear(inputFeaturesCurrent, int(args['MLPLayer0']['number_neurons'])))
        FClayers.append(nn.PReLU(num_parameters=1))
        inputFeaturesCurrent = int(args['MLPLayer0']['number_neurons'])
    if (args['MLPLayer1']['_name'] != "Empty"):
        FClayers.append(nn.Linear(inputFeaturesCurrent, int(args['MLPLayer1']['number_neurons'])))
        FClayers.append(nn.PReLU(num_parameters=1))
        inputFeaturesCurrent = int(args['MLPLayer1']['number_neurons'])
    if (args['MLPLayer2']['_name'] != "Empty"):
        FClayers.append(nn.Linear(inputFeaturesCurrent, int(args['MLPLayer2']['number_neurons'])))
        FClayers.append(nn.PReLU(num_parameters=1))
        inputFeaturesCurrent = int(args['MLPLayer2']['number_neurons'])
    if (args['MLPLayer3']['_name'] != "Empty"):
        FClayers.append(nn.Linear(inputFeaturesCurrent, int(args['MLPLayer3']['number_neurons'])))
        FClayers.append(nn.PReLU(num_parameters=1))
        inputFeaturesCurrent = int(args['MLPLayer3']['number_neurons'])
    FClayers.append(nn.Linear(inputFeaturesCurrent, 1))


	network_regularizer = flop_regularizer.GammaFlopsRegularizer(
		 output_boundary = [pred.op],
		 input_boundary = [x.op, y.op],
		 gamma_threshold = 1e-3
	)

	regularization_strength = 1e-8
	regularizer_loss = (network_regularizer.get_regularization_term() * regularization_strength)

	print(y)

	print(pred)


	#test_lost
	model_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y,1),logits=pred))

	cost_op = network_regularizer.get_cost()
	optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
	train_op = optimizer.minimize(model_loss + regularizer_loss)


	exporter = structure_exporter.StructureExporter(network_regularizer.op_regularizer_manager)
	init = tf.global_variables_initializer()

	morphnet_max_steps = 10
	with tf.Session() as sess:
	  tf.global_variables_initializer().run()
	  for step in range(morphnet_max_steps):
			# sess = tf.Session(graph=tf.get_default_graph())
			_, structure_exporter_tensors = sess.run([train_op, exporter.tensors])
			if (step % 1000 == 0):
				exporter.populate_tensor_values(structure_exporter_tensors)
				exporter.create_file_and_save_alive_counts("train_dir", step)
			
		

    net = NetFix(1, 100, 100, 15)
	
    net.fit()

	
if __name__ == '__main__':
    try:
        
		# Dataset
		dataset = Meta_Dataset()
		dataset.scale = 0.80
	 
	 	 parser = argparse.ArgumentParser()
		parser.add_argument(
		'--use_fp16',
		 default=False,
		 help='Use half floats instead of full floats if True.',
		 action='store_true')
		 parser.add_argument(
		 --self_test',
		 efault=False,
		 action='store_true',
		 help='True if running a self test.')
		 
		 
	if 1 == 1:
            seq_len = dataset.fps * 4 + 25  
            
            trainer = Trainer(dataset, net, '15point_mask5')
			trainer.un  =  unparsed;
            trainer.head_size = 100
            trainer.dataset_step = 1
            trainer.psd_start = 0
            trainer.batch_size = 1
            trainer.batch_seq = seq_len
			
            # Trainer train
            trainer.train_psd_class_r3()
            sys.exit(0)
    except Exception as exception:
        _logger.exception(exception)
        raise
		

  