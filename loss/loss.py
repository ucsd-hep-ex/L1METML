import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

def custom_loss_wrapper(normFac=1, use_symmetry=False, symmetry_weight=1, mse_weight=1):
    '''
    customized loss function to improve the recoil response,
    by balancing the response above one and below one
    '''


    def custom_loss(y_true, y_pred):

        px_truth = K.flatten(y_true[:, 0])
        py_truth = K.flatten(y_true[:, 1])
        px_pred = K.flatten(y_pred[:, 0])
        py_pred = K.flatten(y_pred[:, 1])

        pt_truth = K.sqrt(px_truth*px_truth + py_truth*py_truth)

        #px_truth1 = px_truth / pt_truth
        #py_truth1 = py_truth / pt_truth

        # using absolute response
        # upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred)/pt_truth
        upar_pred = K.sqrt(px_pred * px_pred + py_pred * py_pred) - pt_truth
        pt_cut = pt_truth > 0.
        upar_pred = tf.boolean_mask(upar_pred, pt_cut)
        pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)

        #filter_bin0 = pt_truth_filtered < 50./normFac
        filter_bin0 = tf.logical_and(pt_truth_filtered > 50./normFac,  pt_truth_filtered < 100./normFac)
        filter_bin1 = tf.logical_and(pt_truth_filtered > 100./normFac, pt_truth_filtered < 200./normFac)
        filter_bin2 = tf.logical_and(pt_truth_filtered > 200./normFac, pt_truth_filtered < 300./normFac)
        filter_bin3 = tf.logical_and(pt_truth_filtered > 300./normFac, pt_truth_filtered < 400./normFac)
        filter_bin4 = pt_truth_filtered > 400./normFac

        upar_pred_pos_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred > 0.))
        upar_pred_neg_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred < 0.))
        upar_pred_pos_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred > 0.))
        upar_pred_neg_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred < 0.))
        upar_pred_pos_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred > 0.))
        upar_pred_neg_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred < 0.))
        upar_pred_pos_bin3 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin3, upar_pred > 0.))
        upar_pred_neg_bin3 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin3, upar_pred < 0.))
        upar_pred_pos_bin4 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin4, upar_pred > 0.))
        upar_pred_neg_bin4 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin4, upar_pred < 0.))
        #upar_pred_pos_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred > 0.))
        #upar_pred_neg_bin5 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin5, upar_pred < 0.))
        norm = tf.reduce_sum(pt_truth_filtered)
        dev = tf.abs(tf.reduce_sum(upar_pred_pos_bin0) + tf.reduce_sum(upar_pred_neg_bin0))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin1) + tf.reduce_sum(upar_pred_neg_bin1))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin2) + tf.reduce_sum(upar_pred_neg_bin2))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin3) + tf.reduce_sum(upar_pred_neg_bin3))
        dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin4) + tf.reduce_sum(upar_pred_neg_bin4))
        #dev += tf.abs(tf.reduce_sum(upar_pred_pos_bin5) + tf.reduce_sum(upar_pred_neg_bin5))
        dev /= norm

        mse_loss = mse_weight*0.5*normFac**2*K.mean((px_pred - px_truth)**2 + (py_pred - py_truth)**2)
        base_loss = mse_loss + 7000.*dev
        #loss += 200.*dev
        if use_symmetry:
            # Calculate residuals
            px_residual = px_pred - px_truth
            py_residual = py_pred - py_truth
            
            # Symmetry penalty terms 
            # Mean symmetry 
            px_var = K.mean(px_residual)
            py_var = K.mean(py_residual)
            mean_penalty = K.abs(px_var - py_var)

            # Option to later add variance/std_dev penalty

            # Toal symmetry penalty (add oters here later if needed)
            symmetry_penalty = mean_penalty
            
            # Combine base loss with symmetry penalty
            total_loss = base_loss + symmetry_weight * symmetry_penalty
            return total_loss
        
        else:
            return base_loss
        
        
    return custom_loss
