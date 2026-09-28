We need to participate and win Kaggle competition: 
https://www.kaggle.com/competitions/gemma-4-developer-agent

Competion details are present in:
./documentation/competition_details

You can use kaggle cli api query to get additional details about the competition. Sample commands are 
present in ./notes.md

Initial dataset files can be found here: ./data
Dataset readme: ./data/HARNESS_README.md
Full file list: ./data/files.csv

Sample solutions can be found here: ./documentation/sample_code

The current machine we are running has very low specs so we can't download the whole dataset and run it locally.
You may use the local setup to run some cursory check but do not download the whole dataset.

We do not want to create one big jupyter notebook with all the code.
The code will be developed in multiple python scripts following the coding standards mentioned here:
./documentation/prompts/coding_standards.md
There will a final jupyter notebook whose only job will be to import the python script entry point and run.

The python scripts will be uploaded to Kaggle as dataset and imported there so that the jupyter can import
the files there and run.

The source files will be copied over to ./kaggle_staging and then using the command
kaggle datasets version -p . -m "tag version / details" will be pushed to server. Give a new version number
so that it is easy to traceback to a specific version.

Add detailed login since the run will be primarily on remote server. We will need to use logs to troubleshoot, improve results.

We will keep adding new files here ./documentation/sample_code/high_scorers as other folks submit higher scoring
results. We can then adapt our solution to imbibe there's, however, note: some of the high scoring solutions 
are over-fitted to public dataset / scoring and fail miserably on unseen data. We need to evaluate if taking in the 
new solution is truly worth it.

Python env:
source ~/python_envs/p313_llm/bin/activate

Can you come up with a detailed design to tackle this problem and generate the documentation at:
./documentation/design
