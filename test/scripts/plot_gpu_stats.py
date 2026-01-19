import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

def plot_gpu_stats(csv_file, output_image):
    try:
        # Read the CSV. nvidia-smi puts spaces after commas, so skipinitialspace is needed
        df = pd.read_csv(csv_file, skipinitialspace=True)
    except FileNotFoundError:
        print(f"Error: Could not find log file {csv_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Log file {csv_file} is empty. Did the run finish instantly?")
        sys.exit(1)

    # Cleanup column names (removes ' [%]', ' [MiB]', etc.)
    df.columns = [c.split(' [')[0] for c in df.columns]

    # Cleanup data cells (remove units and convert to float)
    # We use .astype(str) first to handle potential existing strings safely
    if 'utilization.gpu' in df.columns:
        df['utilization.gpu'] = df['utilization.gpu'].astype(str).str.replace(' %', '').astype(float)
    
    if 'memory.used' in df.columns:
        df['memory.used'] = df['memory.used'].astype(str).str.replace(' MiB', '').astype(float)
    
    # Convert timestamp string to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot 1: Utilization (Left Axis - Red)
    color_util = 'tab:red'
    ax1.set_xlabel('Time (HH:MM:SS)')
    ax1.set_ylabel('GPU Utilization (%)', color=color_util)
    ax1.plot(df['timestamp'], df['utilization.gpu'], color=color_util, label='Utilization', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color_util)
    ax1.set_ylim(0, 105) # Fixed scale for %
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot 2: Memory (Right Axis - Blue)
    ax2 = ax1.twinx()
    color_mem = 'tab:blue'
    ax2.set_ylabel('Memory Used (MiB)', color=color_mem)
    ax2.plot(df['timestamp'], df['memory.used'], color=color_mem, label='Memory', linewidth=1.5, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_mem)

    # Format Time Axis
    plt.gcf().autofmt_xdate()
    # Format as Hour:Minute:Second
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.title(f'AlphaFold 3 Run: GPU Metrics\nSource: {csv_file}')
    plt.tight_layout()
    
    plt.savefig(output_image)
    print(f"Success: Plot saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GPU stats from nvidia-smi CSV')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    parser.add_argument('output_image', help='Path to save the output PNG')
    
    args = parser.parse_args()
    plot_gpu_stats(args.csv_file, args.output_image)