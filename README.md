# Birdy Generative Adversarial Network
## Overview
GAN of birdy images.
Crawls through google images for each search term (separated by newline) in ./data/search_terms.txt.
Set train_data folder in main.py train_data argument to folder containing crawled images, then trains GAN on the data.
Outputs generated images in ./data/ folder

## Requirements
Python 3.6, Selenium, pattern

## Run
```bash
main.py [--crawl]  
	[--train_data]  
	[--moden_no]  
	[--batch_size]  
	[--num_epochs]  
	[--lr]  
	[--gradient_acc_steps]  
```

## Results
![](https://github.com/plkmo/Birdy_Generative_adversarial_network/tree/master/results/DLoss.png)
![](https://github.com/plkmo/Birdy_Generative_adversarial_network/tree/master/results/GLoss.png)

![](https://github.com/plkmo/Birdy_Generative_adversarial_network/tree/master/results/birdy_703.png)
Left: Training image, Right: Generated image

