"""
Comprehensive evaluation script for RVC models.

Evaluates a trained RVC model using objective quality metrics.

Usage:
    python testing/evaluate_model.py --model path/to/model.pth --test-dir path/to/test_data --output results.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing.utils.metrics import evaluate_conversion, compare_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_test_pairs(test_dir: str) -> List[Dict[str, str]]:
    """
    Find test audio pairs in directory.
    
    Expected structure:
        test_dir/
            source_001.wav
            target_001.wav
            source_002.wav
            target_002.wav
            ...
    
    Args:
        test_dir: Directory containing test audio pairs
        
    Returns:
        List of dictionaries with 'source' and 'target' paths
    """
    test_dir = Path(test_dir)
    pairs = []
    
    # Find all source files
    source_files = sorted(test_dir.glob("source_*.wav"))
    
    for source_file in source_files:
        # Extract number
        base_name = source_file.stem
        number = base_name.split('_')[1]
        
        # Find corresponding target
        target_file = test_dir / f"target_{number}.wav"
        
        if target_file.exists():
            pairs.append({
                'source': str(source_file),
                'target': str(target_file),
                'id': number
            })
        else:
            logger.warning(f"No target found for {source_file}")
    
    logger.info(f"Found {len(pairs)} test pairs")
    return pairs


def run_inference(model_path: str, source_audio: str, output_path: str) -> str:
    """
    Run RVC inference on source audio.
    
    Args:
        model_path: Path to trained model
        source_audio: Path to source audio
        output_path: Path to save converted audio
        
    Returns:
        Path to converted audio
    """
    # TODO: Implement actual inference
    # For now, this is a placeholder
    logger.info(f"Running inference...")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Source: {source_audio}")
    logger.info(f"  Output: {output_path}")
    
    # In actual implementation, would load model and run conversion
    # For now, just copy source to output as placeholder
    import shutil
    shutil.copy(source_audio, output_path)
    
    return output_path


def evaluate_model(
    model_path: str,
    test_pairs: List[Dict[str, str]],
    output_dir: str
) -> Dict[str, any]:
    """
    Evaluate model on test pairs.
    
    Args:
        model_path: Path to trained model
        test_pairs: List of test audio pairs
        output_dir: Directory to save results
        
    Returns:
        Evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'model': model_path,
        'num_test_pairs': len(test_pairs),
        'per_sample_results': [],
        'aggregate_metrics': {}
    }
    
    # Evaluate each test pair
    for pair in test_pairs:
        logger.info(f"\nEvaluating pair {pair['id']}")
        logger.info("=" * 60)
        
        # Run inference
        converted_path = os.path.join(output_dir, f"converted_{pair['id']}.wav")
        run_inference(model_path, pair['source'], converted_path)
        
        # Evaluate
        metrics = evaluate_conversion(
            converted_audio=converted_path,
            target_audio=pair['target'],
            source_audio=pair['source']
        )
        
        # Store results
        sample_result = {
            'id': pair['id'],
            'source': pair['source'],
            'target': pair['target'],
            'converted': converted_path,
            'metrics': metrics
        }
        results['per_sample_results'].append(sample_result)
    
    # Compute aggregate metrics
    if results['per_sample_results']:
        metrics_keys = results['per_sample_results'][0]['metrics'].keys()
        
        for key in metrics_keys:
            values = [r['metrics'][key] for r in results['per_sample_results'] if key in r['metrics']]
            if values and all(isinstance(v, (int, float)) for v in values):
                results['aggregate_metrics'][key] = {
                    'mean': float(sum(values) / len(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'std': float((sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5)
                }
    
    return results


def print_results(results: Dict[str, any]):
    """Print evaluation results in human-readable format."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nModel: {results['model']}")
    logger.info(f"Test pairs: {results['num_test_pairs']}")
    
    if 'aggregate_metrics' in results and results['aggregate_metrics']:
        logger.info("\nAggregate Metrics:")
        logger.info("-" * 60)
        logger.info(f"{'Metric':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        logger.info("-" * 60)
        
        for metric, stats in results['aggregate_metrics'].items():
            logger.info(
                f"{metric:<30} "
                f"{stats['mean']:<12.3f} "
                f"{stats['std']:<12.3f} "
                f"{stats['min']:<12.3f} "
                f"{stats['max']:<12.3f}"
            )
    
    logger.info("\n" + "=" * 60)


def save_results(results: Dict[str, any], output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")
    
    # Also save markdown summary
    md_file = output_file.replace('.json', '.md')
    with open(md_file, 'w') as f:
        f.write(f"# Evaluation Results\n\n")
        f.write(f"**Model:** `{results['model']}`\n\n")
        f.write(f"**Test Pairs:** {results['num_test_pairs']}\n\n")
        
        if 'aggregate_metrics' in results and results['aggregate_metrics']:
            f.write("## Aggregate Metrics\n\n")
            f.write("| Metric | Mean | Std | Min | Max |\n")
            f.write("|--------|------|-----|-----|-----|\n")
            
            for metric, stats in results['aggregate_metrics'].items():
                f.write(
                    f"| {metric} | "
                    f"{stats['mean']:.3f} | "
                    f"{stats['std']:.3f} | "
                    f"{stats['min']:.3f} | "
                    f"{stats['max']:.3f} |\n"
                )
            f.write("\n")
        
        f.write("## Per-Sample Results\n\n")
        for sample in results['per_sample_results']:
            f.write(f"### Sample {sample['id']}\n\n")
            for metric, value in sample['metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"- **{metric}:** {value:.3f}\n")
            f.write("\n")
    
    logger.info(f"Markdown summary saved to {md_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RVC model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--test-dir", required=True, help="Directory with test audio pairs")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--output-dir", default=None, help="Directory for converted audio (default: next to output file)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return 1
    
    if not os.path.exists(args.test_dir):
        logger.error(f"Test directory not found: {args.test_dir}")
        return 1
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.output),
            "converted_audio"
        )
    
    # Find test pairs
    test_pairs = find_test_pairs(args.test_dir)
    
    if not test_pairs:
        logger.error("No test pairs found!")
        return 1
    
    # Evaluate model
    results = evaluate_model(args.model, test_pairs, args.output_dir)
    
    # Print and save results
    print_results(results)
    save_results(results, args.output)
    
    logger.info("\nEvaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

