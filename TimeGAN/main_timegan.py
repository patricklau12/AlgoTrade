from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import warnings
import pickle
import os

from timegan import timegan
from data_loading import real_data_loading, sine_data_generation
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

warnings.filterwarnings("ignore")

def save_generated_data(data, data_name, parameters):
    """Save generated synthetic data into a specific directory."""
    data_path = './generated_data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    filename = f"{data_name}_module_{parameters['module']}_hidden_{parameters['hidden_dim']}_layers_{parameters['num_layer']}.pkl"
    with open(data_path + filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Generated data saved to {data_path + filename}")

def main(args):
    """Main function for timeGAN experiments."""
    # Data loading
    if args.data_name in ['stock', 'energy']:
        ori_data = real_data_loading(args.data_name, args.seq_len)
    elif args.data_name == 'sine':
        ori_data = sine_data_generation(10000, args.seq_len, 5)
    
    print(args.data_name + ' dataset is ready.')
    
    # Synthetic data generation by TimeGAN
    parameters = {
        'module': args.module,
        'hidden_dim': args.hidden_dim,
        'num_layer': args.num_layer,
        'iterations': args.iteration,
        'batch_size': args.batch_size
    }
    generated_data = timegan(ori_data, parameters)
    print('Finish Synthetic Data Generation')
    
    # Performance metrics
    metric_results = {
        'discriminative': np.mean([discriminative_score_metrics(ori_data, generated_data) for _ in range(args.metric_iteration)]),
        'predictive': np.mean([predictive_score_metrics(ori_data, generated_data) for _ in range(args.metric_iteration)])
    }
    visualization(ori_data, generated_data, 'pca')
    visualization(ori_data, generated_data, 'tsne')
    print(metric_results)
    
    # Save generated data
    save_generated_data(generated_data, args.data_name, parameters)
    
    return ori_data, generated_data, metric_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', choices=['sine', 'stock', 'energy'], default='stock', type=str)
    parser.add_argument('--seq_len', help='sequence length', default=24, type=int)
    parser.add_argument('--module', choices=['gru', 'lstm', 'lstmLN'], default='gru', type=str)
    parser.add_argument('--hidden_dim', help='hidden state dimensions', default=24, type=int)
    parser.add_argument('--num_layer', help='number of layers', default=3, type=int)
    parser.add_argument('--iteration', help='Training iterations', default=50000, type=int)
    parser.add_argument('--batch_size', help='the number of samples in mini-batch', default=128, type=int)
    parser.add_argument('--metric_iteration', help='iterations of the metric computation', default=10, type=int)
    args = parser.parse_args()
    ori_data, generated_data, metrics = main(args)
