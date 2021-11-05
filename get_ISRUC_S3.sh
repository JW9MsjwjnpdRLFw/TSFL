mkdir -p ./data/ISRUC_S3/ExtractedChannels
mkdir -p ./data/ISRUC_S3/RawData
echo 'Make data dir: ./data/ISRUC_S3'

cd ./data/ISRUC_S3/RawData
for s in {1..10}:
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$s.rar
    unrar x $s.rar
done
echo 'Download Data to "./data/ISRUC_S3/RawData" complete.'

cd ./data/ISRUC_S3/ExtractedChannels
for s in {1..10}:
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject$s.mat
done
echo 'Download ExtractedChannels to "./data/ISRUC_S3/ExtractedChannels" complete.'
