"""
src/deepfake_detector/__main__.py
===================================
Student ID : 21396816
Purpose    : Entry-point shim enabling the package to be invoked as:

                python -m deepfake_detector --input clip.mp4 --out result.json

             This is the form required by the project brief. All logic lives
             in scripts/run_inference.py — this file is deliberately two lines
             so there is exactly one copy of the CLI.
"""

from scripts.run_inference import main

main()