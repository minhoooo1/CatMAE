wget -P ./data https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip ./data/DAVIS-2017-trainval-480p.zip -d ./data
cp -r data/DAVIS data/DAVIS_480_880
python convert_480_880.py