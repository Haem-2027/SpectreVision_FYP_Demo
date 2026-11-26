SPECTREVISION - FYP DEMO
========================

RUN THE WEBSITE
  1. Install Python 3.10 - 3.12 from python.org (tick "Add Python to PATH" during install).
  2. Unzip this folder anywhere.
  3. Double-click Start_SpectreVision.bat  (first time it installs Flask automatically).
  4. The browser opens http://127.0.0.1:5000
       Username: spectrevision
       Password: SV-demo-2026
  5. Keep the black window open while presenting. Close it to stop the site.

CONTENTS
  app.py                  the website (Flask). Users are stored in users.db (SQLite, hashed passwords).
  videos/                 the processed demo videos shown on the site
  models/                 box_loading.pt (loading), helmet_best.pt (helmet), yolov8s.pt (people)
  code/Loading_Counter.py loading / unloading counter (box + person-carrying-box, line crossing)
  code/Helmet_Pipeline.py helmet detection (find people first, then check each head)

RE-RUN THE DETECTION (optional, needs a GPU ideally)
  pip install -r requirements.txt
  cd code
  python Loading_Counter.py --source "path\to\IP28_Loading - Trim.mp4"
  python Helmet_Pipeline.py --source "path\to\PPE DETECTION FOOTAGE.mp4"
