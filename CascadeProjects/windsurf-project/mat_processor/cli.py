"""
Command-line interface for the MAT file processor.
"""
import argparse
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

from .io import load_mat, explore_mat_structure
from .analysis import parameterize_data, convert_to_dataframe
from .visualization import MATVisualizer


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description='Process and analyze .mat files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information about a .mat file')
    info_parser.add_argument('file', help='Path to the .mat file')
    info_parser.add_argument('--format', choices=['text', 'json'], default='text',
                           help='Output format')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Create plots from .mat file data')
    plot_parser.add_argument('file', help='Path to the .mat file')
    plot_parser.add_argument('--output-dir', default='.', 
                           help='Directory to save plots')
    plot_parser.add_argument('--dpi', type=int, default=300,
                           help='DPI for output figures')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', 
                                         help='Convert .mat file to other formats')
    convert_parser.add_argument('file', help='Path to the .mat file')
    convert_parser.add_argument('--format', choices=['csv', 'json', 'hdf5'], 
                              default='csv', help='Output format')
    convert_parser.add_argument('--output', '-o', 
                              help='Output file or directory (default: same as input with new extension)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'info':
            data = load_mat(args.file)
            if args.format == 'json':
                # Convert numpy arrays to lists for JSON serialization
                serializable_data = {}
                for k, v in data.items():
                    if isinstance(v, (np.ndarray, np.generic)):
                        serializable_data[k] = v.tolist()
                    else:
                        serializable_data[k] = v
                print(json.dumps(serializable_data, indent=2))
            else:
                print(f"File: {args.file}")
                print("-" * 50)
                print(explore_mat_structure(data))
                
        elif args.command == 'plot':
            data = load_mat(args.file)
            visualizer = MATVisualizer(data)
            
            # Create output directory if it doesn't exist
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Simple auto-plotting of numeric arrays
            for key, value in data.items():
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                    if value.ndim == 1:
                        # 1D data: histogram
                        fig = visualizer.plot_histogram(key)
                        fig.savefig(output_dir / f"{key}_hist.png", dpi=args.dpi)
                        plt.close(fig)
                    elif value.ndim == 2 and value.shape[0] > 1 and value.shape[1] > 1:
                        # 2D data: heatmap
                        fig = visualizer.plot_heatmap(key)
                        fig.savefig(output_dir / f"{key}_heatmap.png", dpi=args.dpi)
                        plt.close(fig)
            
            print(f"Plots saved to {output_dir.absolute()}")
            
        elif args.command == 'convert':
            data = load_mat(args.file)
            output_path = args.output or Path(args.file).with_suffix(f'.{args.format}')
            
            if args.format == 'csv':
                # Convert all 1D and 2D arrays to DataFrames and save as CSV
                dfs = convert_to_dataframe(data)
                if not dfs:
                    print("No convertible data found")
                    return
                    
                output_path = Path(output_path)
                output_path.mkdir(parents=True, exist_ok=True)
                
                for name, df in dfs.items():
                    safe_name = "".join(c if c.isalnum() else "_" for c in name)
                    df.to_csv(output_path / f"{safe_name}.csv", index=False)
                
                print(f"CSV files saved to {output_path.absolute()}")
                
            elif args.format == 'json':
                # Convert numpy arrays to lists for JSON serialization
                serializable_data = {}
                for k, v in data.items():
                    if isinstance(v, (np.ndarray, np.generic)):
                        serializable_data[k] = v.tolist()
                    else:
                        serializable_data[k] = v
                
                with open(output_path, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
                
                print(f"JSON file saved to {output_path}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
