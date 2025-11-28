#!/bin/bash
pip3 install --no-cache-dir -r requirements.txt
python3 -m uvicorn app:app --host 0.0.0.0 --port 8080