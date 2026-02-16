import subprocess
import time
import os
import sys

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    start_time = time.time()
    
    script_path = os.path.join("src", script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found.")
        return False
        
    try:
        # Run the script using the same python interpreter
        result = subprocess.run([sys.executable, script_path], check=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"✓ {script_name} completed in {duration:.2f}s\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with exit code {e.returncode}\n")
        return False

def main():
    if not os.path.exists('output_analysis'):
        os.makedirs('output_analysis')
        
    scripts = [
        "analyze_velocity.py",
        "analyze_energy.py",
        "analyze_mixing.py",
        "analyze_vorticity.py",
        "analyze_spectra.py",
        "analyze_density_pdf.py"
    ]
    
    print("========================================")
    print("Starting Comprehensive RTI Analysis")
    print("========================================\n")
    
    total_start = time.time()
    success_count = 0
    
    for script in scripts:
        if run_script(script):
            success_count += 1
            
    total_end = time.time()
    total_duration = total_end - total_start
    
    print("========================================")
    print(f"Analysis Complete in {total_duration:.2f}s")
    print(f"Success: {success_count}/{len(scripts)}")
    print(f"Results saved to: {os.path.abspath('output_analysis')}")
    print("========================================")

if __name__ == "__main__":
    main()
