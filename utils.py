import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_input(path):
		file_list_gen = glob('{}*targ*.npy'.format(path))
		file_list_MET = glob('{}*MET*.npy'.format(path))
		file_list_MET_xy = glob('{}*MET*xy*.npy'.format(path))
		file_list_Pup = glob('{}*Pup*.npy'.format(path))

		target_array_xy = np.load('{}'.format(file_list_gen[0]))
		feature_MET_array = np.load('{}'.format(file_list_MET[0]))
		feature_MET_array_xy = np.load('{}'.format(file_list_MET_xy[0]))
		feature_pupcandi_array_xy = np.load('{}'.format(file_list_Pup[0]))

		return target_array_xy, feature_MET_array, feature_MET_array_xy, feature_pupcandi_array_xy


def custom_loss(y_true, y_pred):
    '''
    cutmoized loss function to improve the recoil response,
    by balancing the response above one and below one
    '''
    import keras.backend as K
    import tensorflow as tf

    px_truth = K.flatten(y_true[:,0])
    py_truth = K.flatten(y_true[:,1])
    px_pred = K.flatten(y_pred[:,0])
    py_pred = K.flatten(y_pred[:,1])

    pt_truth = K.sqrt(px_truth*px_truth + py_truth*py_truth)

    px_truth1 = px_truth / pt_truth
    py_truth1 = py_truth / pt_truth

    # using absolute response
    # upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred)/pt_truth
    upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred) - pt_truth
    pt_cut = pt_truth > 0./50.
    upar_pred = tf.boolean_mask(upar_pred, pt_cut)
    pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)

    
    '''
    filter_bin0 = pt_truth_filtered < 5./50.
    filter_bin1 = tf.logical_and(pt_truth_filtered > 5./50., pt_truth_filtered < 10./50.)
    filter_bin2 = pt_truth_filtered > 10./50.
    '''
    filter_bin0 = pt_truth_filtered < 20.
    filter_bin1 = tf.logical_and(pt_truth_filtered > 20., pt_truth_filtered < 50.)
    filter_bin2 = pt_truth_filtered > 50.

    upar_pred_pos_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred > 0.))
    upar_pred_neg_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred < 0.))
    upar_pred_pos_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred > 0.))
    upar_pred_neg_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred < 0.))
    upar_pred_pos_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred > 0.))
    upar_pred_neg_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred < 0.))
    norm = tf.reduce_sum(pt_truth_filtered)
    dev = tf.abs(tf.reduce_sum(upar_pred_pos_bin0) + tf.reduce_sum(upar_pred_neg_bin0))
    dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin1) + tf.reduce_sum(upar_pred_neg_bin1))
    dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin2) + tf.reduce_sum(upar_pred_neg_bin2))
    dev /= norm

    loss = 0.5*K.mean((px_pred - px_truth)**2 + (py_pred - py_truth)**2)

    #loss += 200.*dev
    loss += 10.*dev
    return loss


def flatting(GenMET):
    bin_width = 5
    min_ = 0
    max_ = 1000

    bin_number = int((max_ - min_)/bin_width)
    bin_ = np.linspace(0,300, num = bin_number)

    MET_interval = np.histogram(GenMET[:,0], bin_)

    bin_indices = np.digitize(GenMET[:,0], bin_)

    bin_size = np.zeros(bin_number)
    mask = [True]

    for i in range(GenMET.shape[0]):
        for j in range(bin_number):
            if (bin_width*(j + 0) + min_ <= GenMET[i,0] < bin_width*(j + 1) + min_):
                bin_size[j] = bin_size[j] + 1
                if bin_size[j] >= 3000:
                    mask.append(bool(False))
                else:
                    if (i == 0):
                        1
                    else:
                        mask.append(bool(True))

    GenMET = GenMET[mask]
    plt.hist(GenMET[:,0], bins=np.linspace(0, 500, 100))
    plt.savefig('flat.png')
    plt.show()

    return mask
