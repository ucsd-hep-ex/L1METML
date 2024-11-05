import argparse
import tensorflow
from models import dense_embedding
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import hls4ml
import pandas as pd
from qkeras.utils import _add_supported_quantized_objects
from models import dense_embedding, dense_embedding_quantized
from utils import preProcessing
import h5py
import scipy
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

co = {}
_add_supported_quantized_objects(co)

def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def load_model(model_name):
    if 'quantized' in model_name:
        model = tensorflow.keras.models.load_model(f'models/baseline_DeepMET_quantized/{model_name}.h5', compile=False, custom_objects=co)
    else:
        model = tensorflow.keras.models.load_model(f'models/baseline_DeepMET/{model_name}.h5', compile=False)
    return model

def configure_hls_model(model, config_params):
    config = hls4ml.utils.config_from_keras_model(model, 
                                                granularity='name',
                                                default_reuse_factor=config_params['reuse_factor'],
                                                default_precision=config_params['precision'])
    config['Model']['Strategy'] = config_params['strategy']
    for name in config['LayerName'].keys():
        config['LayerName'][name]['Trace'] = config_params['trace']
    config['LayerName']['input_cat0']['Precision']['result'] = 'ap_uint<4>'
    config['LayerName']['input_cat1']['Precision']['result'] = 'ap_uint<4>'
    # config['LayerName']['input_cont']['Precision']['result'] = 'ap_fixed<20,10>'
    #if 'q_dense' in config['LayerName']:
    #    config['LayerName']['q_dense']['Precision']['accum'] = 'ap_fixed<32,16>'
    #    config['LayerName']['q_dense']['Precision']['weight'] = 'ap_fixed<32,16>'
    #    config['LayerName']['q_dense']['Precision']['bias'] = 'ap_fixed<32,16>'
    # config['LayerName']['q_dense_1']['Precision']['accum'] = 'ap_fixed<32,16>'
    # config['LayerName']['q_dense_1']['Precision']['weight'] = 'ap_fixed<32,16>'
    # config['LayerName']['q_dense_1']['Precision']['bias'] = 'ap_fixed<32,16>'
    config['LayerName']['multiply']['n_elem'] = 100
    config['LayerName']['output']['n_filt'] = 2
    # skip optimize_pointwise_conv
    # config['SkipOptimizers'] = ['optimize_pointwise_conv']
    # for layer in config['LayerName'].keys():
    #    config['LayerName'][layer]['Trace'] = True

    print("-----------------------------------")
    print_dict(config)
    return config

def convert_to_hls_model(model, config, ouput_dir, io_type, part, clock_period, project_name):
    print("-----------------------------------")
    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                        hls_config=config,
                                                        io_type=io_type,
                                                        output_dir=output_dir,
                                                        part=part,
                                                        clock_period=clock_period,
                                                        project_name=project_name,
    )
    hls_model.compile()
    return hls_model

def preprocess_data(file_path, norm_factor):
    with h5py.File(file_path, 'r') as f:
        # 1000 test events is good enough
        X = f['X'][:1000]
        y = -f['Y'][:1000]
    X_preprocessed = list(preProcessing(X, normFac=norm_factor))
    return [np.ascontiguousarray(x) for x in X_preprocessed], y


def plot_metrics(data_to_plot, hls_model, model, output_dir):
    met = data_to_plot['met']
    met_pred = data_to_plot['met_pred']
    met_hls = data_to_plot['met_hls']
    met_pup = data_to_plot['met_pup']
    met_pup_x = data_to_plot['met_pup_x']
    met_pup_y = data_to_plot['met_pup_y']
    y_pred = data_to_plot['y_pred']
    y = data_to_plot['y']
    X_pre = data_to_plot['x_pre']
    
    
    df = pd.DataFrame.from_dict({
            'Gen MET': met,
            'PUPPI MET': met_pup,
            'QKeras MET': met_pred,
            'hls4ml MET': met_hls,
            })
    plt.figure()
    seaborn.pairplot(df, corner=True)
    plt.savefig(f'{output_dir}/profiling_MET.png', dpi=300)
    plt.close()

    df = pd.DataFrame.from_dict(
            {'Gen MET x': y[:, 0], 
            'PUPPI MET x': met_pup_x, 
            'QKeras MET x': y_pred[:, 0], 
            'hls4ml MET x': y_hls[:, 0],
            })
    plt.figure()
    seaborn.pairplot(df, corner=True)
    plt.savefig(f'{output_dir}/profiling_MET_x.png', dpi=300)

    df = pd.DataFrame.from_dict({
            'Gen MET y': y[:, 1], 
            'PUPPI MET y': met_pup_y, 
            'QKeras MET y': y_pred[:, 1], 
            'hls4ml MET y': y_hls[:, 1]
            })
    plt.figure()
    seaborn.pairplot(df, corner=True)
    plt.savefig(f'{output_dir}/profiling_MET_y.png', dpi=300)

    response_pup = met_pup / met
    response_pred = met_pred / met
    response_hls = met_hls / met
    bins = np.linspace(0, 2, 25)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.hist(response_pup, bins=bins, label=f'PUPPI, median={np.median(response_pup):0.2f}, IQR={scipy.stats.iqr(response_pup):0.2f}')
    plt.legend()
    plt.xlabel("MET response $\hat{y}/y$")
    plt.ylabel("Events")
    plt.subplot(1, 3, 2)
    plt.hist(response_pred, bins=bins, label=f'QKeras, median={np.median(response_pred):0.2f}, IQR={scipy.stats.iqr(response_pred):0.2f}')
    plt.legend()
    plt.xlabel("MET response $\hat{y}/y$")
    plt.ylabel("Events")
    plt.subplot(1, 3, 3)
    plt.hist(response_hls, bins=bins, label=f'hls4ml, median={np.median(response_hls):0.2f}, IQR={scipy.stats.iqr(response_hls):0.2f}')
    plt.legend()
    plt.xlabel("MET response $\hat{y}/y$")
    plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_MET.png", dpi=300)

    y_hls, hls4ml_trace = hls_model.trace(X_pre)
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_pre)

    for layer in hls4ml_trace.keys():
        plt.figure()
        if layer not in keras_trace: continue
        plt.scatter(hls4ml_trace[layer].flatten(), keras_trace[layer].flatten(), s=0.2)
        min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[layer]))
        max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[layer]))
        plt.plot([min_x, max_x], [min_x, max_x], c='gray')
        plt.xlabel(f'hls4ml {layer}')
        plt.ylabel(f'QKeras {layer}')
        plt.savefig(f'{output_dir}/profiling_{layer}.png', dpi=300)

def main(args):
    model_name = args.model_name

    model = load_model('trained_DeepMET')

    config_params = {
        'reuse-factor': 1, 
        'strategy': 'Latency', 
        'precision': 'ap_fixed<32,16>', 
        'trace': True,
        }
    reuse_factor = 1
    precision = 'ap_fixed<32,16>'
    io_type = 'io_parallel'
    strategy = 'Latency'
    output_dir = 'hls_output_{}_{}_{}_rf{}_{}'.format(model_name ,io_type, strategy, reuse_factor, precision)
    batch_size = 1
    synth = False
    trace = True
    normFac = 1

    # check everthing works
    model.summary()
    model.save('{}/model.h5'.format(output_dir))

    config = configure_hls_model(model, {})

    hls_model = convert_to_hls_model(model, config, output_dir, io_type, 'xcvu13p-flga2577-2-e', 5, 'L1METML_v1')

    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='{}/model_hls4ml.png'.format(output_dir))

    if synth:
        hls_model.build(synth=synth)
        hls4ml.report.read_vivado_report(output_dir)


    X_pre, y = preprocess_data('data/test_data.h5', norm_factor=1)



    y_pred = model.predict(X_pre)
    y_hls = hls_model.predict(X_pre)

    met = np.hypot(y[:, 0], y[:, 1])
    met_pred = np.hypot(y_pred[:, 0], y_pred[:, 1]) * normFac
    met_hls = np.hypot(y_hls[:, 0], y_hls[:, 1]) * normFac
    met_pup_x = np.sum(X[:, :, 1], axis=-1) #does this need to be X_pre?
    met_pup_y = np.sum(X[:, :, 2], axis=-1) #does this need to be X_pre?
    met_pup = np.hypot(met_pup_x, met_pup_y)

    data_to_plot = {
        'met': met,
        'met_pred': met_pred,
        'met_hls': met_hls,
        'met_pup': met_pup,
        'met_pup_x': met_pup_x,
        'met_pup_y': met_pup_y,
        'y_pred': y_pred,
        'y': y,
        'x_pre': X_pre,
    }

    plot_metrics(data_to_plot, hls_model, model, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='trained_DeepMET',
        choices = [
            'trained_DeepMET',
            'trained_quantized_DeepMET',
            'trained_quantized_DeepMET_normfac1000'
            ],
        help='Model name')

    args = parser.parse_args()
    
    main(args)

