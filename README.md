# sonic-subterfuge

## Install required modules

Run `pip install -r .\code\requirements.txt` to install all necessary modules. Created with Python 3.11.

## Download IRMAS dataset

Run [DownloadIRMAS.ipynb](code/DownloadIRMAS.ipynb) to download and extract the [IRMAS](https://www.upf.edu/web/mtg/irmas) dataset automatically.

## Genetic Algorithm

To run the genetic algorithm on any of the pre-trained models, use [GeneticAlgo.ipynb](/code/GeneticAlgo.ipynb).

The model and standard scaler can be selected in the third module, and the input and output can be selected like this:

```python
songname = "PATH_TO_INPUT_FILE"
orig_y, sr = librosa.load(songname, sr=44100)
orig_y = np.real(librosa.istft(np.real(librosa.stft(orig_y))))
print("Probability of being what it actually is:", score_prob_of_being(orig_y, sr, songname.split("/")[-2]))
tricked_class = "DESIRED_OUTPUT_CLASS"
print("Probability of being what we want it to be:", score_prob_of_being(orig_y, sr, tricked_class))
```

# [Explanation and Findings](https://r2beeaton.github.io/sonic-subterfuge/)