
- Test a trained model named PongDeterministic-v4.dat in the folder trained_models/I2A/
	python3 main.py --env PongDeterministic-v4 --workers 0 --amsgrad True --gpu-ids 0 --save-score-level 1000 --load True --load-model-dir trained_models/I2A/

	--gpu-ids 0 # use one gpu, if empty no gpu is used
	--workers 0 # only test the model no workers for training
	--load True # load an already trained model

- Train a trained model named PongDeterministic-v4.dat in the folder trained_models/I2A/
	python3 main.py --env PongDeterministic-v4 --workers 3 --amsgrad True --gpu-ids 0 --save-score-level 1000
	--workers 3 # a3c uses 3 workers to train the model

- Train an environment model
	python3 main_environment_model.py

	expect that you have an gpu availaible if not set use_cuda=False in the main_environment_model.py file

- Visualize Imagination Core
	python3 main_imagination_core.py
	
	expect that you have an gpu availaible if not set use_cuda=False in the main_environment_model.py file
	visualize the pong imagination core, by loading pretrained policy and environment models
	
- Environment Model Standalone
	In addition we added the Environment Model Standalone, which we used to explore different Architectures of the environment. You can find the code we used for that in the 

*Please Note* We due to size constraints we removed the pretrained models and datasets. In case you want to run the code please contact us.

*Sources:* We based our code on the repository https://github.com/dgriff777/rl_a3c_pytorch, which implements A3C for Reinforcement Learning.
