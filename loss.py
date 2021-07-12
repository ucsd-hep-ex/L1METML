def custom_loss(y_true, y_pred):
    '''
    cutmoized loss function to improve the recoil response,
    by balancing the response above one and below one
    '''
    import tensorflow.keras.backend as K
    import tensorflow as tf

    px_truth = K.flatten(y_true[:,0])
    py_truth = K.flatten(y_true[:,1])
    px_pred = K.flatten(y_pred[:,0])
    py_pred = K.flatten(y_pred[:,1])

    pt_truth = K.sqrt(px_truth*px_truth + py_truth*py_truth)

    #px_truth1 = px_truth / pt_truth
    #py_truth1 = py_truth / pt_truth

    # using absolute response
    # upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred)/pt_truth
    upar_pred = K.sqrt(px_pred * px_pred + py_pred * py_pred) - pt_truth
    pt_cut = pt_truth > 0.
    upar_pred = tf.boolean_mask(upar_pred, pt_cut)
    pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)

    #filter_bin0 = pt_truth_filtered < 50.
    filter_bin0 = tf.logical_and(pt_truth_filtered > 50.,  pt_truth_filtered < 100.)
    filter_bin1 = tf.logical_and(pt_truth_filtered > 100., pt_truth_filtered < 200.)
    filter_bin2 = tf.logical_and(pt_truth_filtered > 200., pt_truth_filtered < 300.)
    filter_bin3 = tf.logical_and(pt_truth_filtered > 300., pt_truth_filtered < 400.)
    filter_bin4 = pt_truth_filtered > 400.

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

    loss = 0.5*K.mean((px_pred - px_truth)**2 + (py_pred - py_truth)**2)

    #loss += 200.*dev
    loss += 100.*dev
    return loss
