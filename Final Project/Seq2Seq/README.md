# Seq2Seq Model For Training Telegram Data

The files ``responseData.txt`` and ``responseDict.npy`` have been pre-generated. 

To run the Seq2Seq algorithm, navigate to the directory where this folder is stored in using Terminal/Command Prompt and run the following lines of code: 

## Installing Tensorflow
`sudo pip install tensorflow==1.14.0 -t /opt/venv/lib/python3.7/site-packages/ --upgrade`

## Running Dict2Wordlist.py --> this will give you a wordList.txt file
`python3 Dict2Wordlist.py`

## Running Seq2Seq.py
`python3 Seq2Seq.py`

From this, you will get the training matrices `Seq2SeqXTrain.npy` and `Seq2SeqYTrain.npy` as well as the `.ckpt` files in a new folder called `models`.

The pre-trained models from my previous training is in the folder `pre-trained-models`, all except the main data file (pretrained_seq2seq.ckpt-50000.data-00000-of-00001) - because of Minerva Forum's 50MB file upload limit, I will be storing the main data file (281.6MB) on Google Drive, which you can download here: https://drive.google.com/file/d/1VCdLvGEnZ3gH5Zpt5X5cvH81xFwBWAEa/view?usp=sharing

---

# Chatterbot Implementation

The chatterbot implementation is located in the `chatterbot.ipynb` Python notebook.

Unfortunately, you won't be able to run that notebook because I won't be including the unprocessed corpus of my Telegram chat history in this assignment submission for confidentiality purposes.