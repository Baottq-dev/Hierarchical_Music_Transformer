#!/usr/bin/env python3
"""
Test Module Runner - Tests model performance and evaluates generated music
"""

import argparse
import sys
import os
import glob
from source.test import ModelEvaluator, EvaluationMetrics, ModelTester

def main():
    parser = argparse.ArgumentParser(description="Test model performance and evaluate generated music")
    parser.add_argument("--model_path", help="Path to trained model")
    parser.add_argument("--generated_files", nargs="+", help="Generated MIDI files to evaluate")
    parser.add_argument("--reference_files", nargs="+", help="Reference MIDI files for comparison")
    parser.add_argument("--output_dir", default="test_results", help="Output directory")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive testing")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples for benchmark")
    
    args = parser.parse_args()
    
    print("🧪 Starting Model Testing...")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    metrics = EvaluationMetrics()
    
    # Step 1: Test model loading (if model provided)
    if args.model_path:
        print(f"\n🔧 Step 1: Testing model loading...")
        tester = ModelTester(args.model_path)
        load_result = tester.test_model_loading()
        
        if load_result['success']:
            print("✅ Model loaded successfully")
            if load_result['model_info']:
                print(f"  Parameters: {load_result['model_info']['total_parameters']:,}")
        else:
            print(f"❌ Model loading failed: {load_result['error']}")
            return
    
    # Step 2: Evaluate generated files
    if args.generated_files:
        print(f"\n📊 Step 2: Evaluating generated files...")
        
        # Check if files exist
        valid_files = []
        for file in args.generated_files:
            if os.path.exists(file):
                valid_files.append(file)
            else:
                print(f"⚠️ File not found: {file}")
        
        if valid_files:
            evaluation_results = evaluator.evaluate_batch(
                generated_files=valid_files,
                reference_files=args.reference_files
            )
            
            # Save evaluation report
            report_file = os.path.join(args.output_dir, "evaluation_report.json")
            evaluator.generate_evaluation_report(evaluation_results, report_file)
            
            # Generate plots
            evaluator.plot_metrics(evaluation_results, args.output_dir)
            
            print(f"✅ Evaluated {len(valid_files)} generated files")
        else:
            print("❌ No valid generated files found")
    
    # Step 3: Run comprehensive testing
    if args.comprehensive and args.model_path:
        print(f"\n🔍 Step 3: Running comprehensive testing...")
        comprehensive_results = tester.run_comprehensive_test(args.output_dir)
        
        if comprehensive_results['pipeline_integration']['success']:
            print("✅ Comprehensive testing completed successfully")
        else:
            print("❌ Comprehensive testing failed")
            for error in comprehensive_results['pipeline_integration'].get('errors', []):
                print(f"  - {error}")
    
    # Step 4: Run performance benchmark
    if args.benchmark and args.model_path:
        print(f"\n⚡ Step 4: Running performance benchmark...")
        benchmark_results = tester.benchmark_performance(
            num_samples=args.num_samples
        )
        
        if benchmark_results['success']:
            print("✅ Performance benchmark completed")
            print(f"  Average generation time: {benchmark_results.get('avg_generation_time', 0):.2f}s")
            print(f"  Average sequence length: {benchmark_results.get('avg_sequence_length', 0):.1f}")
        else:
            print("❌ Performance benchmark failed")
    
    # Step 5: Calculate detailed metrics
    if args.generated_files:
        print(f"\n📈 Step 5: Calculating detailed metrics...")
        
        detailed_metrics = []
        for file in args.generated_files:
            if os.path.exists(file):
                try:
                    import pretty_midi
                    midi_data = pretty_midi.PrettyMIDI(file)
                    metrics_result = metrics.calculate_all_metrics(midi_data)
                    detailed_metrics.append({
                        'file': file,
                        'metrics': metrics_result
                    })
                except Exception as e:
                    print(f"⚠️ Error calculating metrics for {file}: {e}")
        
        if detailed_metrics:
            # Save detailed metrics
            metrics_file = os.path.join(args.output_dir, "detailed_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(detailed_metrics, f, indent=2, ensure_ascii=False)
            
            # Print summary
            print(f"✅ Calculated detailed metrics for {len(detailed_metrics)} files")
            
            # Calculate averages
            avg_metrics = {}
            metric_names = detailed_metrics[0]['metrics'].keys()
            
            for metric_name in metric_names:
                values = [m['metrics'].get(metric_name, 0) for m in detailed_metrics 
                         if isinstance(m['metrics'].get(metric_name, 0), (int, float))]
                if values:
                    avg_metrics[f'{metric_name}_avg'] = sum(values) / len(values)
                    avg_metrics[f'{metric_name}_min'] = min(values)
                    avg_metrics[f'{metric_name}_max'] = max(values)
            
            print(f"\n📊 Average Metrics:")
            for key, value in avg_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
    
    # Step 6: Compare with reference files
    if args.generated_files and args.reference_files:
        print(f"\n🔄 Step 6: Comparing with reference files...")
        
        comparisons = []
        for gen_file, ref_file in zip(args.generated_files, args.reference_files):
            if os.path.exists(gen_file) and os.path.exists(ref_file):
                try:
                    comparison = evaluator.evaluate_generated_vs_reference(gen_file, ref_file)
                    if comparison:
                        comparisons.append({
                            'generated_file': gen_file,
                            'reference_file': ref_file,
                            'comparison': comparison
                        })
                except Exception as e:
                    print(f"⚠️ Error comparing {gen_file} with {ref_file}: {e}")
        
        if comparisons:
            # Save comparisons
            comparison_file = os.path.join(args.output_dir, "comparisons.json")
            with open(comparison_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(comparisons, f, indent=2, ensure_ascii=False)
            
            # Print summary
            similarities = [c['comparison']['overall_similarity'] for c in comparisons]
            avg_similarity = sum(similarities) / len(similarities)
            
            print(f"✅ Compared {len(comparisons)} file pairs")
            print(f"  Average similarity: {avg_similarity:.3f}")
            print(f"  Min similarity: {min(similarities):.3f}")
            print(f"  Max similarity: {max(similarities):.3f}")
    
    print(f"\n🎉 Testing completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    # List output files
    output_files = glob.glob(os.path.join(args.output_dir, "*"))
    if output_files:
        print(f"\n📄 Generated files:")
        for file in output_files:
            print(f"  - {os.path.basename(file)}")

if __name__ == "__main__":
    main() 