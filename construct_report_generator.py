"""
construct_report_generator.py
Helper entry point to trigger the IEEE report suite generator.
Delegates execution to scripts/generate_ieee_report.py.

Authors:
  Heet Gujarati (202451069), Yash Jagani (202451077), Brahmesh Italiya (202451038)
  Department of Computer Science and Engineering
  Indian Institute of Information Technology, Vadodara, India
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.generate_ieee_report import main

if __name__ == "__main__":
    main()
