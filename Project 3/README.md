# Instructions for Running Code

1. Create virtual environment and install packages

`python3 -m venv venv`
`./venv/bin/activate`
`pip -r install requirements.txt`

2. Fetch input data for character transformer

`wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

3. Convert python scripts to notebooks

`./convert_to_notebook.sh`

4. Run code

`python3 recurrent_neural_network.py`
`python3 transformer.py`
`python3 vae.py`