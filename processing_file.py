import os
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pyannote.audio import Model
import whisper
import numpy as np
import gc
from pyannote.audio import Inference
from numpy.linalg import norm
from pyannote.core import Segment
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def read(k):
    y = np.array(k.get_array_of_samples())
    return np.float32(y) / 32768

def millisec(timeStr):
    spl = timeStr.split(":")
    return (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)

def identify_speaker(embedding1,inference):
    #go through each speaker file and check best speaker
    voice_files=os.listdir('people_voice_data/')
    max=0
    name_m=''
    for vf in voice_files:
        embedding2 = inference(f"people_voice_data/{vf}")
        cosine = np.dot(embedding1,embedding2)/(norm(embedding1)*norm(embedding2))
        print('\t',vf[:-4],': ',cosine)
        if round(cosine,1)>max:
            max=cosine
            name_m=vf
    if round(max,1)>=0.3:
        return name_m[:-4]
    else:
        return None

def speaker_identification(segs_list,fp, inference):
    #speakers with their respective segments stored in dictionary
    speaker_segs=defaultdict(list)
    for l in range(len(segs_list)):
        #split the above l element in k: ['[', '00:00:00.008', '-->', '', '00:00:08.106]', 'A', 'SPEAKER_04']
        j = segs_list[l].split(" ")
        #take out start and end of the segment in millisec for that specific speaker, as read supports millisec
        start = int(millisec(j[1]))
        end = int(millisec(j[4][:-1]))
        speaker_segs[j[6]].append(j[1]+'-->'+j[4][:-1])
    #take average embeddings of speaker segments
    avg_speaker_embs=defaultdict(list)
    for sp in speaker_segs.keys():
        for sp_seg in speaker_segs[sp]:
            try:
                start=millisec(sp_seg.split('-->')[0])/1000
                end=millisec(sp_seg.split('-->')[1])/1000
                excerpt = Segment(start,end)
                emb1 = inference.crop(fp, excerpt)
                try:
                    array1 = np.array(emb1)
                    array2 = np.array(avg_speaker_embs[sp])
                    average_array = (array1+array2)/2
                    average_list = average_array.tolist()
                    avg_speaker_embs[sp]=average_list
                except:
                    avg_speaker_embs[sp]=emb1
            except:
                print(sp,"'s Segment: ",sp_seg," Not Used for Speaker Recognition")
    #identify speakers with embeddings/average embeddings
    res={}
    for sp in avg_speaker_embs:
        print(sp,'=>')
        speaker=identify_speaker(avg_speaker_embs[sp],inference)
        print(sp,':',speaker)
        if speaker!=None:
            res[sp]=speaker
    return res

def do_work(filepath,nspeakers=None):
    #------------------------DIARIZATION ACCORDING TO DIFFERENT SPEAKERS------------------------
    #pyannote pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
    #output for each speaker, one element in 'k': '[ 00:00:00.008 -->  00:00:08.106] A SPEAKER_04'
    if nspeakers!=None:
        k=str(pipeline(filepath,num_speakers=nspeakers)).split('\n')
    else:
        k=str(pipeline(filepath)).split('\n')
    print(k)
    del pipeline
    gc.collect()
    #segmenting audio for sample rate of 16KB
    audio = AudioSegment.from_mp3(filepath)
    audio = audio.set_frame_rate(16000)
    #delete older output file
    if os.path.exists('tr_file.txt'):
        os.remove('tr_file.txt')
    #------------------------SPEAKER IDENTIFICATION------------------------
    #embedding model
    emb_model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")
    inference = Inference(emb_model, window="whole")
    identified_speakers=speaker_identification(k,filepath,inference)
    #------------------------TRANSCRIPTIONS AND SAVING RESULTS------------------------
    #load whisper model
    model = whisper.load_model("medium.en") #only english allowed
    for l in range(len(k)):
        print('Doing work on: ',l+1,'/',len(k))
        #split the above l element in k: ['[', '00:00:00.008', '-->', '', '00:00:08.106]', 'A', 'SPEAKER_04']
        j = k[l].split(" ")
        #take out start and end of the segment in millisec for that specific speaker, as read supports millisec
        start = int(millisec(j[1]))
        end = int(millisec(j[4][:-1]))
        #read that specific segment
        tr = read(audio[start:end])
        #identify speaker
        if j[6] in identified_speakers.keys():
            j[6]=identified_speakers[j[6]]
        else: #not found by average so lets try by clips
            try:
                excerpt = Segment(start/1000,end/1000)
                emb1 = inference.crop(filepath, excerpt)
                sp=identify_speaker(emb1,inference)
                if sp!=None:
                    identified_speakers[j[6]]=sp
                    j[6]=sp
            except:
                print('--------')
        #transcribe that read audio segment
        result = model.transcribe(tr, fp16=False)
        #write to the file
        f = open("tr_file.txt", "a")
        f.write(f'\n[ {j[1]} -- {j[3]} ] {j[6]} : {result["text"]}')
        f.close()
        #for memory efficiency
        del f
        del result
        del tr
        del j
        gc.collect()
    print('Final Identified Speakers: \n',identified_speakers)
    return 0