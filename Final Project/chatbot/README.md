# Seq2Seq Chatbot Trained On Telegram Messages Corpus On Flask Server

Note: Because of Github's 100MB file upload limit, I couldn’t include the main data file (`pretrained_seq2seq.ckpt-50000.data-00000-of-00001`, which had a file size of 281.6MB). Therefore, I will be storing it on Google Drive, which you can download here: https://drive.google.com/file/d/1VCdLvGEnZ3gH5Zpt5X5cvH81xFwBWAEa/view?usp=sharing. Remember, you will have to download it into the ``models`` folder with the exact name `pretrained_seq2seq.ckpt-50000.data-00000-of-00001` for the following steps to work.

To launch the webapp, navigate to the directory where this folder is stored in using Terminal/Command Prompt and run the following lines of code: 

## Installing the required libraries
`pip install -r requirements.txt`

## Running the python webapp
`python3 app.py`
