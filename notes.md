### Kaggle API Commands:
Getting competition overview:  
```bash
kaggle competitions pages list -c gemma-4-developer-agent --content  
```
  
Pulling code file:  
```bash
kaggle kernels pull ryanholbrook/getting-started-gemma-4-developer-agent  
```
  
Code files list:  
```bash
kaggle competitions files -c gemma-4-developer-agent --page-size 600 --format csv > files.csv  
kaggle competitions files -c gemma-4-developer-agent --page-size 600 --page-token "<next page token>" --format csv > files_2.csv
```

Specific code file:  
```bash  
kaggle competitions download -c gemma-4-developer-agent -f HARNESS_README.md
```

Kaggle dataset:
```bash
kaggle datasets init
# After editing metadata:
kaggle datasets create -p .
kaggle datasets version -p . -m "tag version / details"
```