# Gated-Attention Architectures for Task-Oriented Language Grounding
This is a PyTorch implementation of the AAAI-18 paper:

[Gated-Attention Architectures for Task-Oriented Language Grounding](https://arxiv.org/abs/1706.07230)<br />
Devendra Singh Chaplot, Kanthashree Mysore Sathyendra, Rama Kumar Pasumarthi, Dheeraj Rajagopal, Ruslan Salakhutdinov<br />
Carnegie Mellon University

Project Website: https://sites.google.com/view/gated-attention

![example](./docs/example.gif)

### This repository contains:
- Code for training an A3C-LSTM agent using Gated-Attention
- Code for Doom-based language grounding environment

## Dependencies
- [ViZDoom](https://github.com/mwydmuch/ViZDoom)
- [PyTorch](http://pytorch.org)
- Opencv 

(We recommend using [Anaconda](https://www.anaconda.com/download))

## Usage

### Using the Environment
For running a random agent:
```
python env_test.py
```
To play in the environment:
```
python env_test.py --interactive 1
```
To change the difficulty of the environment (easy/medium/hard):
```
python env_test.py -d easy
```

### Training Gated-Attention A3C-LSTM agent
For training a A3C-LSTM agent with 32 threads:
```
python a3c_main.py --num-processes 32 --evaluate 0
```
The code will save the best model at `./saved/model_best`.

To the test the pre-trained model for Multitask Generalization:
```
python a3c_main.py --evaluate 1 --load saved/pretrained_model
```
To the test the pre-trained model for Zero-shot Task Generalization:
```
python a3c_main.py --evaluate 2 --load saved/pretrained_model
``` 
To the visualize the model while testing add '--visualize 1':<br />
```
python a3c_main.py --evaluate 2 --load saved/pretrained_model --visualize 1
``` 
To test the trained model, use `--load saved/model_best` in the above commands.

All arguments for a3c_main.py:
```
  -h, --help            show this help message and exit
  -l MAX_EPISODE_LENGTH, --max-episode-length MAX_EPISODE_LENGTH
                        maximum length of an episode (default: 30)
  -d DIFFICULTY, --difficulty DIFFICULTY
                        Difficulty of the environment, "easy", "medium" or
                        "hard" (default: hard)
  --living-reward LIVING_REWARD
                        Default reward at each time step (default: 0, change
                        to -0.005 to encourage shorter paths)
  --frame-width FRAME_WIDTH
                        Frame width (default: 300)
  --frame-height FRAME_HEIGHT
                        Frame height (default: 168)
  -v VISUALIZE, --visualize VISUALIZE
                        Visualize the envrionment (default: 0, use 0 for
                        faster training)
  --sleep SLEEP         Sleep between frames for better visualization
                        (default: 0)
  --scenario-path SCENARIO_PATH
                        Doom scenario file to load (default: maps/room.wad)
  --interactive INTERACTIVE
                        Interactive mode enables human to play (default: 0)
  --all-instr-file ALL_INSTR_FILE
                        All instructions file (default:
                        data/instructions_all.json)
  --train-instr-file TRAIN_INSTR_FILE
                        Train instructions file (default:
                        data/instructions_train.json)
  --test-instr-file TEST_INSTR_FILE
                        Test instructions file (default:
                        data/instructions_test.json)
  --object-size-file OBJECT_SIZE_FILE
                        Object size file (default: data/object_sizes.txt)
  --lr LR               learning rate (default: 0.001)
  --gamma G             discount factor for rewards (default: 0.99)
  --tau T               parameter for GAE (default: 1.00)
  --seed S              random seed (default: 1)
  -n N, --num-processes N
                        how many training processes to use (default: 4)
  --num-steps NS        number of forward steps in A3C (default: 20)
  --load LOAD           model path to load, 0 to not reload (default: 0)
  -e EVALUATE, --evaluate EVALUATE
                        0:Train, 1:Evaluate MultiTask Generalization
                        2:Evaluate Zero-shot Generalization (default: 0)
  --dump-location DUMP_LOCATION
                        path to dump models and log (default: ./saved/)
```

## Demostration videos:
Multitask Generalization video: https://www.youtube.com/watch?v=YJG8fwkv7gA

Zero-shot Task Generalization video: https://www.youtube.com/watch?v=JziCKsLrudE

Different stages of training: https://www.youtube.com/watch?v=o_G6was03N0

## Cite as
>Chaplot, D.S., Sathyendra, K.M., Pasumarthi, R.K., Rajagopal, D. and Salakhutdinov, R., 2017. Gated-Attention Architectures for Task-Oriented Language Grounding. arXiv preprint arXiv:1706.07230. ([PDF](http://arxiv.org/abs/1706.07230))

### Bibtex:
```
@article{chaplot2017gated,
  title={Gated-Attention Architectures for Task-Oriented Language Grounding},
  author={Chaplot, Devendra Singh and Sathyendra, Kanthashree Mysore and Pasumarthi, Rama Kumar and Rajagopal, Dheeraj and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:1706.07230},
  year={2017}
}
```

## Acknowledgements
This repository uses ViZDoom API (https://github.com/mwydmuch/ViZDoom) and parts of the code from the API. The implementation of A3C is borrowed from https://github.com/ikostrikov/pytorch-a3c. The poisson-disc code is borrowed from https://github.com/IHautaI/poisson-disc.
