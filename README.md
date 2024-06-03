# Speaker-Identification-Audio-Transcription
Audio File Processing Information and Code with three approaches:
We are using the **Second Approach** from the colab file.
In this project, we are providing Audio File's transcription with Multi-Speaker Identification using two different models, as mentioned in *Google Colab File*! 
Why Three Approaches for Speaker Identification?
It is because, as per my testcases and results, the better results greatly depend upon the nature of audio file. If the accent of the speaker remains consistent, then 1st approach is best. If we feel that the speaker's voice is deviating and some collisions a little bit, then 2nd approach is best. However, if we feel that there are a lot of deviations and disturbances in segment, then 3rd approach is good. These insights greatly depend on my intuitions and tests; therefore, the results can vary. **Please share your insights as well.**
In this repoistory, we are providing **Tkinter** interface using Python. The code is available in `project_updated.ipynb`. The file_processing has the same code as 2nd approach, which is used to process audio files selected by user dynamically. 
My computer's specifications: 

![image](https://github.com/bnkf1156f/Speaker-Identification-Audio-Transcription/assets/96035737/44e0eef1-dca3-4b12-b01f-51ff3dcc7ea4)

**Note:**
Make sure to execute **Folderlock.bat**, and password: virkhase6a. Now you will get to see "Private Folder", in which login_dets.txt used for login information at the beginning, and for output saving, there is each registered person's folder, where the files requested by the user "to save" are saved.
Also, Please download whole repoistory!!

**Product Scope**
Provide a system which can recognize different speakers by their voice features in a recorded audio file. 
Allow accurate attribution of dialogue to specific speakers in an audio.
Enable the use of OPENAI Whisper, pre-trained model, to accurately transcribe audio to textual format.
Allow formatting of transcribed text into a dialogue format, clearly indicating the exchange of dialogue between different speakers.

**References**
Whisper Github Repository for Speech to Text function: https://github.com/openai/whisper
Pyannote Github Repository for Speaker Diarization Function:     https://github.com/pyannote/pyannote-audio
Python tkinter library for User Interface: https://docs.python.org/3/library/tkinter.html
Github 'speaker-diarization' repoistories: https://github.com/topics/speaker-diarization

**Project Working:**

![image](https://github.com/bnkf1156f/Speaker-Identification-Audio-Transcription/assets/96035737/6e491b9f-af07-4f09-a331-a5a6d0f68d8d)
![image](https://github.com/bnkf1156f/Speaker-Identification-Audio-Transcription/assets/96035737/140aaecb-f7c0-4097-b653-69ad44af6676)
![image](https://github.com/bnkf1156f/Speaker-Identification-Audio-Transcription/assets/96035737/02ec85b9-328e-47e7-afd3-331abf155f75)
![image](https://github.com/bnkf1156f/Speaker-Identification-Audio-Transcription/assets/96035737/1d61819a-6b98-4c83-ad70-c3f855b60096)

Now after correct file selection, click **start processing..**
Then go to .ipynb file output, and there you will find the processing details as shown in colab. After last segment processing, "save output.." if required.
For **File Retrieval** from Private Folder:
Click **Back To Options**, and select Retrieve Old Files. Then will be shown page:

![image](https://github.com/bnkf1156f/Speaker-Identification-Audio-Transcription/assets/96035737/58530af5-33d7-4706-aa11-601fa18e93eb)

Here you will only be able to see your (particular logged in user) files only.

You are free to mingle the code, and change according to your own requirements.
**If there are any questions, then please let me know!**
