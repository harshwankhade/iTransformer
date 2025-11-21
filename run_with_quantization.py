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


def main():
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
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

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

                    # ============================================================
                    # STEP 1: PRINT ORIGINAL MODEL ANALYSIS
                    # ============================================================
                    print("\n" + "-"*80)
                    print("ORIGINAL MODEL ANALYSIS")
                    print("-"*80)
                    quant.print_model_analysis(original_model)

                    # ============================================================
                    # STEP 2: APPLY QUANTIZATION
                    # ============================================================
                    print("\n" + "-"*80)
                    print("QUANTIZATION PROCESS")
                    print("-"*80)

                    if args.quantization_type == 'dynamic':
                        quantized_model = quant.quantize_model_dynamic(original_model)
                    else:
                        quantized_model = quant.quantize_model_static(original_model, test_loader)

                    print("[3/4] Quantization complete")

                    # ============================================================
                    # STEP 3: PROFILE BOTH MODELS
                    # ============================================================
                    print("\n" + "-"*80)
                    print("PERFORMANCE PROFILING")
                    print("-"*80)

                    # Get a test batch for profiling
                    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))

                    # Profile original model
                    print("\nProfiling ORIGINAL model...")
                    original_metrics = quant.profile_inference(
                        original_model,
                        batch_x, batch_x_mark,
                        torch.zeros_like(batch_y[:, -args.pred_len:, :]),
                        batch_y_mark,
                        device=str(device),
                        num_iterations=args.profile_iterations
                    )

                    # Profile quantized model (move to CPU for inference)
                    print("Profiling QUANTIZED model...")
                    quantized_metrics = quant.profile_inference(
                        quantized_model,
                        batch_x, batch_x_mark,
                        torch.zeros_like(batch_y[:, -args.pred_len:, :]),
                        batch_y_mark,
                        device='cpu',
                        num_iterations=args.profile_iterations
                    )

                    print("[4/4] Profiling complete")

                    # ============================================================
                    # STEP 4: DISPLAY COMPARISON RESULTS
                    # ============================================================
                    print("\n" + "="*80)
                    print("COMPREHENSIVE COMPARISON RESULTS")
                    print("="*80)

                    from utils.quantization import get_model_size, compare_models

                    # Calculate metrics
                    speedup = original_metrics['elapsed_time_per_iteration'] / quantized_metrics['elapsed_time_per_iteration']
                    orig_size = get_model_size(original_model)
                    quant_size = get_model_size(quantized_model)
                    model_comparison = compare_models(original_model, quantized_model)

                    # Memory improvement
                    memory_improvement = 0
                    if original_metrics['memory_delta_mb'] != 0:
                        memory_improvement = ((original_metrics['memory_delta_mb'] - quantized_metrics['memory_delta_mb']) 
                                            / abs(original_metrics['memory_delta_mb']) * 100)

                    # Display results
                    print("\n📊 TIMING ANALYSIS")
                    print("="*80)
                    print(f"Original Model - Time per Iteration: {original_metrics['elapsed_time_per_iteration']*1000:.4f} ms")
                    print(f"Quantized Model - Time per Iteration: {quantized_metrics['elapsed_time_per_iteration']*1000:.4f} ms")
                    print(f"Speedup Factor: {speedup:.2f}x")
                    print(f"Time Savings: {(1-1/speedup)*100:.2f}%")

                    print("\n💾 MEMORY ANALYSIS")
                    print("="*80)
                    print(f"Original Model - Memory Delta: {original_metrics['memory_delta_mb']:+.2f} MB")
                    print(f"Quantized Model - Memory Delta: {quantized_metrics['memory_delta_mb']:+.2f} MB")
                    print(f"Memory Improvement: {memory_improvement:.2f}%")

                    if 'gpu_memory_allocated_mb' in original_metrics:
                        print(f"\nGPU Memory (Original): {original_metrics.get('gpu_memory_allocated_mb', 0):.2f} MB")
                        print(f"GPU Memory (Quantized): {quantized_metrics.get('gpu_memory_allocated_mb', 0):.2f} MB")

                    print("\n📦 MODEL SIZE ANALYSIS")
                    print("="*80)
                    print(f"Original Model Size: {orig_size['total_size_mb']:.4f} MB")
                    print(f"Quantized Model Size: {quant_size['total_size_mb']:.4f} MB")
                    print(f"Compression Ratio: {model_comparison['compression_ratio']:.2f}x")
                    print(f"Size Reduction: {model_comparison['size_reduction_percent']:.2f}%")

                    print("\n⚡ SUMMARY")
                    print("="*80)
                    print(f"Quantization Type: {args.quantization_type.upper()}")
                    print(f"Speedup: {speedup:.2f}x faster")
                    print(f"Size: {model_comparison['compression_ratio']:.2f}x smaller")
                    print(f"Memory: {memory_improvement:.2f}% saved")
                    print(f"\n✅ Training + Quantization Complete!")
                    print("="*80)

                    # Save quantized model
                    quant.save_quantized_model(f'./checkpoints/{setting}/quantized_model.pth')
                    print(f"\n💾 Quantized model saved to: ./checkpoints/{setting}/quantized_model.pth")

                except Exception as e:
                    print(f"\n❌ Error during quantization: {str(e)}")
                    import traceback
                    traceback.print_exc()

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
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
