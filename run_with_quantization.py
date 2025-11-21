"""
Enhanced training script with quantization and profiling.
Trains the model, then applies quantization, and displays improvements.
"""

import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_quantized import Exp_Quantized_Long_Term_Forecast
from utils.profiling import PerformanceProfiler
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
    
    def print_summary(self):
        """Print summary of all tracked events."""
        if not self.events:
            return
            
        total_time = sum(e['time_seconds'] for e in self.events)
        max_cpu_memory = max(e['end_memory_mb'] for e in self.events)
        max_gpu_memory = max(e['peak_gpu_memory_mb'] for e in self.events)
        total_memory_delta = sum(e['memory_delta_mb'] for e in self.events)
        total_gpu_memory_delta = sum(e['gpu_memory_delta_mb'] for e in self.events)
        
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # 📊 EXECUTION STATISTICS
        print(f"\n📊 EXECUTION STATISTICS")
        print(f"{'─'*80}")
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Peak CPU Memory: {max_cpu_memory:.2f} MB")
        if torch.cuda.is_available():
            print(f"Peak GPU Memory: {max_gpu_memory:.2f} MB")
        
        # 📈 DETAILED BREAKDOWN
        print(f"\n📈 DETAILED BREAKDOWN")
        print(f"{'─'*80}")
        if torch.cuda.is_available():
            for event in self.events:
                percentage = (event['time_seconds'] / total_time * 100) if total_time > 0 else 0
                print(f"{event['name']:<20} {event['time_seconds']:<10.2f}s  {event['memory_delta_mb']:+>10.2f} MB CPU  {event['gpu_memory_delta_mb']:+>10.2f} MB GPU  {percentage:>6.1f}%")
        else:
            for event in self.events:
                percentage = (event['time_seconds'] / total_time * 100) if total_time > 0 else 0
                print(f"{event['name']:<20} {event['time_seconds']:<10.2f}s  {event['memory_delta_mb']:+>10.2f} MB CPU  {percentage:>6.1f}%")
        
        # ⏱️  TIMING METRICS
        print(f"\n⏱️  TIMING METRICS")
        print(f"{'─'*80}")
        for i, event in enumerate(self.events, 1):
            print(f"[{i}] {event['name']:<50} {event['time_seconds']:>10.2f}s ({event['time_seconds']/60:>8.2f}m)")
        
        # 💾 MEMORY METRICS
        print(f"\n💾 MEMORY METRICS")
        print(f"{'─'*80}")
        print(f"Total CPU Memory Change: {total_memory_delta:+.2f} MB")
        print(f"Average Memory per Event: {total_memory_delta/len(self.events):+.2f} MB")
        print(f"Peak System Memory: {max_cpu_memory:.2f} MB")
        
        # 🔋 GPU MEMORY METRICS
        if torch.cuda.is_available():
            print(f"\n🔋 GPU MEMORY METRICS")
            print(f"{'─'*80}")
            print(f"Total GPU Memory Change: {total_gpu_memory_delta:+.2f} MB")
            print(f"Peak GPU Memory: {max_gpu_memory:.2f} MB")
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
            print(f"GPU Memory Cached: {torch.cuda.memory_cached()/1024/1024:.2f} MB")
        
        # Final Summary
        print(f"\n{'─'*80}")
        print(f"✅ Execution completed in {total_time/60:.2f} minutes")
        print(f"✅ Peak memory usage: {max_cpu_memory:.2f} MB (CPU)")
        if torch.cuda.is_available():
            print(f"✅ Peak GPU memory: {max_gpu_memory:.2f} MB")
        print(f"{'─'*80}")


def main():
    # Initialize performance tracker
    tracker = PerformanceTracker()
    
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer with Quantization')

    # Basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='iTransformer',
                        help='model name')

    # Data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # Optimization
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
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training')

    # Quantization options
    parser.add_argument('--apply_quantization', type=bool, default=True, help='apply quantization after training')
    parser.add_argument('--quantization_type', type=str, default='dynamic', help='dynamic or static')
    parser.add_argument('--profile_iterations', type=int, default=10, help='number of iterations for profiling')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # =====================================================================
    # PHASE 1: TRAINING
    # =====================================================================
    print("\n" + "="*80)
    print("PHASE 1: MODEL TRAINING")
    print("="*80)

    if args.is_training:
        for ii in range(args.itr):
            # Setting record of experiments
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

            exp = Exp_Long_Term_Forecast(args)
            
            # Track training
            tracker.start('Training')
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            tracker.stop('Training')

            # Track testing
            tracker.start('Testing')
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            tracker.stop('Testing')

            if args.do_predict:
                tracker.start('Predicting')
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)
                tracker.stop('Predicting')

            torch.cuda.empty_cache()

            # ================================================================
            # PHASE 2: QUANTIZATION AND PROFILING
            # ================================================================
            if args.apply_quantization:
                print("\n" + "="*80)
                print("PHASE 2: QUANTIZATION AND PROFILING")
                print("="*80)

                try:
                    # Initialize quantization helper
                    quant = Exp_Quantized_Long_Term_Forecast()
                    device = exp.device

                    # Get original model
                    original_model = exp.model
                    print("\n[1/4] Original model loaded")

                    # Get test data for profiling
                    _, test_loader = exp._get_data('test')
                    print("[2/4] Test data loaded")

                    # Apply quantization
                    print("\n[3/4] Applying quantization...")
                    if args.quantization_type == 'dynamic':
                        quantized_model = quant.quantize_model_dynamic(original_model)
                    else:
                        quantized_model = quant.quantize_model_static(original_model, test_loader)

                    print("[4/4] Quantization complete")

                    # Get a test batch for profiling
                    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))

                    # Profile original model
                    original_metrics = quant.profile_inference(
                        original_model,
                        batch_x, batch_x_mark,
                        torch.zeros_like(batch_y[:, -args.pred_len:, :]),
                        batch_y_mark,
                        device=str(device),
                        num_iterations=args.profile_iterations
                    )

                    # Profile quantized model (move to CPU for inference)
                    quantized_metrics = quant.profile_inference(
                        quantized_model,
                        batch_x, batch_x_mark,
                        torch.zeros_like(batch_y[:, -args.pred_len:, :]),
                        batch_y_mark,
                        device='cpu',
                        num_iterations=args.profile_iterations
                    )

                    # Save quantized model
                    quant.save_quantized_model(f'./checkpoints/{setting}/quantized_model.pth')
                    print(f"\n✅ Quantized model saved to: ./checkpoints/{setting}/quantized_model.pth")

                except Exception as e:
                    print(f"\n❌ Error during quantization: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Print performance summary
        tracker.print_summary()

    else:
        # Testing only mode
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

        exp = Exp_Long_Term_Forecast(args)
        tracker.start('Testing')
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        tracker.stop('Testing')
        torch.cuda.empty_cache()
        
        tracker.print_summary()


if __name__ == '__main__':
    main()
