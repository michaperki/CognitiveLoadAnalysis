
#!/usr/bin/env python
"""
Experiment Runner for Cognitive Load Analysis v11

This script wraps the complete pipeline process.
"""

import argparse
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cognitive load analysis experiment (v11)")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--data_path", type=str, help="Path to data file (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--skip_validation", action="store_true", help="Skip extensive validation")
    args = parser.parse_args()
    
    exit(main())
