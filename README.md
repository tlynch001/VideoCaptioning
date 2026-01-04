Steps
1) create the wav file with this command line argument: ffmpeg -i caption1.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le audio.wav
2) run the python file (python has to be installed on your pc of course): python capcut_phrase_highlight_stable.py audio.wav
3) burn the captions back into your video: ffmpeg -i caption1.mp4 -vf "ass=captions_phrase.ass" -c:a copy caption1_phrase.mp4
