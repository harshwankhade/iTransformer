import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
import time
import psutil
import os
from datetime import datetime


class PerformanceTracker:
    """Track time and memory usage during training and testing."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None
        self.events = []
        
    def start(self, event_name):
        """Start tracking an event."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            self.start_gpu_memory = 0
            
        print(f"\n⏱️  Starting: {event_name}")
        print(f"   Memory at start: {self.start_memory:.2f} MB")
        if torch.cuda.is_available():
            print(f"   GPU Memory at start: {self.start_gpu_memory:.2f} MB")
        
    def stop(self, event_name):
        """Stop tracking and record event."""
        elapsed_time = time.time() - self.start_time
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - self.start_memory
        
        gpu_memory_delta = 0
        peak_gpu_memory = 0
        if torch.cuda.is_available():
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory_delta = current_gpu_memory - self.start_gpu_memory
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        event_info = {
            'name': event_name,
            'time_seconds': elapsed_time,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'memory_delta_mb': memory_delta,
            'gpu_memory_delta_mb': gpu_memory_delta,
            'peak_gpu_memory_mb': peak_gpu_memory,
            'timestamp': datetime.now()
        }
        self.events.append(event_info)
        
        print(f"✅ Completed: {event_name}")
        print(f"   ⏱️  Time elapsed: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
        print(f"   💾 Memory at end: {end_memory:.2f} MB (Δ {memory_delta:+.2f} MB)")
        if torch.cuda.is_available():
            print(f"   🔋 GPU Memory: {current_gpu_memory:.2f} MB (Δ {gpu_memory_delta:+.2f} MB, Peak: {peak_gpu_memory:.2f} MB)")
        
    def print_summary(self):
        """Print summary of all tracked events."""
        if not self.events:
            return
            
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # Calculate metrics
        total_time = sum(e['time_seconds'] for e in self.events)
        max_cpu_memory = max(e['end_memory_mb'] for e in self.events)
        max_gpu_memory = max(e['peak_gpu_memory_mb'] for e in self.events)
        total_memory_delta = sum(e['memory_delta_mb'] for e in self.events)
        avg_memory_per_event = total_memory_delta / len(self.events)
        
        # Find training and testing times
        training_time = sum(e['time_seconds'] for e in self.events if 'Training' in e['name'])
        testing_time = sum(e['time_seconds'] for e in self.events if 'Testing' in e['name'])
        
        print(f"\n⏱️  TIME METRICS")
        print(f"{'─'*80}")
        print(f"Total Training Time: {training_time:.2f}s ({training_time/60:.2f}m)")
        print(f"Total Testing Time: {testing_time:.2f}s ({testing_time/60:.2f}m)")
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
        
        print(f"\n� MEMORY METRICS")
        print(f"{'─'*80}")
        print(f"Peak CPU Memory: {max_cpu_memory:.2f} MB")
        if torch.cuda.is_available():
            print(f"Peak GPU Memory: {max_gpu_memory:.2f} MB")
        print(f"Average Memory per Event: {avg_memory_per_event:+.2f} MB")
        print(f"{'─'*80}")


if __name__ == '__main__':
    # Initialize performance tracker
    tracker = PerformanceTracker()
    
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            
            # Track training
            tracker.start(f'Training: {setting}')
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            tracker.stop(f'Training: {setting}')

            # Track testing
            tracker.start(f'Testing: {setting}')
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            test_results = exp.test(setting)
            tracker.stop(f'Testing: {setting}')
            
            # Inference speed benchmarking
            print("\n" + "="*80)
            print("INFERENCE SPEEDUP METRICS")
            print("="*80)
            
            # Warm-up runs
            print("\n🔥 Warming up model...")
            exp.model.eval()
            device = torch.device('cuda' if args.use_gpu else 'cpu')
            dummy_input = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(device)
            
            for _ in range(10):
                with torch.no_grad():
                    _ = exp.model(dummy_input, None, None, None)
            
            # Benchmark inference speed
            print("⚡ Benchmarking inference speed...")
            num_iterations = 100
            
            if torch.cuda.is_available() and args.use_gpu:
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = exp.model(dummy_input, None, None, None)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
            else:
                start_time = time.time()
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = exp.model(dummy_input, None, None, None)
                elapsed_time = time.time() - start_time
            
            avg_inference_time = (elapsed_time / num_iterations) * 1000  # Convert to ms
            throughput = num_iterations / elapsed_time  # iterations per second
            
            # Model size calculation
            model_size = sum(p.numel() * p.element_size() for p in exp.model.parameters()) / (1024 * 1024)  # MB
            param_count = sum(p.numel() for p in exp.model.parameters())
            
            print(f"\n⚡ INFERENCE SPEED METRICS")
            print(f"{'─'*80}")
            print(f"Device: {'GPU' if (torch.cuda.is_available() and args.use_gpu) else 'CPU'}")
            print(f"Batch Size: {args.batch_size}")
            print(f"Sequence Length: {args.seq_len}")
            print(f"Number of Iterations: {num_iterations}")
            print(f"Average Inference Time: {avg_inference_time:.4f} ms/iteration")
            print(f"Throughput: {throughput:.2f} iterations/second")
            print(f"Samples/Second: {throughput * args.batch_size:.2f}")
            
            print(f"\n📦 MODEL SIZE METRICS")
            print(f"{'─'*80}")
            print(f"Total Parameters: {param_count:,}")
            print(f"Model Size: {model_size:.2f} MB")
            print(f"Memory per Parameter: {model_size / (param_count / 1e6):.2f} bytes/param")
            print(f"{'─'*80}")

            if args.do_predict:
                tracker.start(f'Predicting: {setting}')
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)
                tracker.stop(f'Predicting: {setting}')

            torch.cuda.empty_cache()
        
        # Print performance summary
        tracker.print_summary()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        
        # Track testing
        tracker.start(f'Testing: {setting}')
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        tracker.stop(f'Testing: {setting}')
        
        torch.cuda.empty_cache()
        
        # Print performance summary
        tracker.print_summary()
