
# Uncomment to install and config kaggle key
#pip3 install kaggle -q
#mkdir ~/.kaggle
#echo '{"username":"ferocha","key":"KAGGLE_KEY_HERE"}' > ~/.kaggle/kaggle.json

# download and extract data
kaggle competitions download -c outbrain-click-prediction -f clicks_train.csv.zip -w
kaggle competitions download -c outbrain-click-prediction -f events.csv.zip -w
kaggle competitions download -c outbrain-click-prediction -f promoted_content.csv.zip -w 
kaggle competitions download -c outbrain-click-prediction -f documents_categories.csv.zip -w
kaggle competitions download -c outbrain-click-prediction -f documents_topics.csv.zip -w
kaggle competitions download -c outbrain-click-prediction -f documents_meta.csv.zip -w
unzip -q "*.zip"