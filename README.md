This is a project to locate former manufactured gas plant (FMGP) sites using historic Sanborn fire insurance maps.

This repository includes a set of Python scripts to:

1. Download Sanborn map scans using the Library of Congress API (LOC_API.py);
2. Analyze scans for circular features that correspond to former MGP sites (detect_circles_LOC.py); and
3. Read output images into a dataframe for analysis (files_to_dataframe.py)
