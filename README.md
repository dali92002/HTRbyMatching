# HTRbyMatching

## Description

A pytorch implementation of the paper [A Few-shot Learning Approach for Historical Ciphered Manuscript Recognition](https://arxiv.org/abs/2009.12577) and its extension [Few Shots Are All You Need: A Progressive Few Shot Learning Approach for Low Resource Handwriting Recognition](https://arxiv.org/abs/2107.10064). The proposed model can be used for low resource handwriting recognition in a few-shot learning scenario. 

<img src="./imgs/model.png"  alt="1" width = 1200px height = 300px >

<img src="./imgs/progressive.png"  alt="1" width = 1200px height = 300px >

## Download Code
clone the repository:
```bash
git clone https://github.com/dali92002/HTRbyMatching
cd HTRbyMatching
```
## Requirements

Create your evironment, with the file htrmatching.yml

## Models

Download the desired weights. These are following the training that was done in [A Few-shot Learning Approach for Historical Ciphered Manuscript Recognition](https://arxiv.org/abs/2009.12577).


<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Training Dataset</th>
    <th class="tg-c3ow">Fine Tuning Dataset</th>
    <th class="tg-c3ow">URL</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" >Omniglot</td>
    <td class="tg-c3ow">  -- </td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/113X6gzFHTIkHZ3XYbyTcCWpQ?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" >Omniglot</td>
    <td class="tg-c3ow">  Borg </td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/113X6gzFHTIkHZ3XYbyTcCWQzAs/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
    <tr>
    <td class="tg-c3ow" >Omniglot</td>
    <td class="tg-c3ow">  Copiale </td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/113X6gzFHTIkHZ3XYbyTcCWpQGV8Qiew?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
    <tr>
    <td class="tg-c3ow" >Omniglot</td>
    <td class="tg-c3ow">  Vatican </td>
    <td class="tg-c3ow"><a href="https://drive.google.com/file/d/113X6gzFHTIkHZ3XYQQzAs/view?usp=sharing" target="_blank" rel="noopener noreferrer">model</a></td>
  </tr>
  
</tbody>
</table>

## Training 

Coming soon ...

## Testing

Download the desired pretrained weigts from the section Models. Then run the following command. Here we are choosing to recognize the lines of the cipher "borg", in a 1 shot scenario, with the model finetuned on the borg (as will be stated in the testing model path). We specify the input data path: lines and alphabet. As well ass the desired output path, here we want in in a floder named output in this same directory and the threshold 0.4. 

```bash
python test.py --cipher borg --testing_model models/borg.pth  --lines ./lines --alphabet ./alphabet  --output ./output_result --shots 1 --thresh 0.4
```

Please, check the folders named lines and alphabet to realize how you should provide your input data. After running you will receive the results in a 3 subfolders of your output folder.  

## Training in a progressive way (Coming soon)

This part is related to the paper [Few Shots Are All You Need: A Progressive Few Shot Learning Approach for Low Resource Handwriting Recognition](https://arxiv.org/abs/2107.10064). ... 

## Citation 
If you find this useful for your research, please cite it as follows:

```bash
python test.py --cipher borg --lines ./lines --alphabet ./alphabet  --output ./output --shots 1 --thresh 0.4
```
```bash
python test.py --cipher borg --lines ./lines --alphabet ./alphabet  --output ./output --shots 1 --thresh 0.4
```