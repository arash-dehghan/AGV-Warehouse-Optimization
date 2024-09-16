# Dynamic AGV Task Allocation in Intelligent Warehouses

This repository provides complementary code and data for the article [Dynamic AGV Task Allocation in Intelligent Warehouses](https://arxiv.org/pdf/2312.16026).

This paper explores the integration of Automated Guided Vehicles (AGVs) in warehouse order picking, a crucial and cost-intensive aspect of warehouse operations. The booming AGV industry, accelerated by the COVID-19 pandemic, is witnessing widespread adoption due to its efficiency, reliability, and cost-effectiveness in automating warehouse tasks. This paper focuses on enhancing the picker-to-parts system, prevalent in small to medium-sized warehouses, through the strategic use of AGVs. We discuss the benefits and applications of AGVs in various warehouse tasks, highlighting their transformative potential in improving operational efficiency. We examine the deployment of AGVs by leading companies in the industry, showcasing their varied functionalities in warehouse management. Addressing the gap in research on optimizing operational performance in hybrid environments where humans and AGVs coexist, our study delves into a dynamic picker-to-parts warehouse scenario. We propose a novel approach Neural Approximate Dynamic Programming approach for coordinating a mixed team of human and AGV workers, aiming to maximize order throughput and operational efficiency. This involves innovative solutions for nonmyopic decision making, order batching, and battery management. We also discuss the integration of advanced robotics technology in automating the complete order-picking process. Through a comprehensive numerical study, our work offers valuable insights for managing a heterogeneous workforce in a hybrid warehouse setting, contributing significantly to the field of warehouse automation and logistics.

## Installation and Execution Setup
In the following, I will describe how to setup a conda environment, initialise python, download CPLEX, and install necessary packages for setting up the repository in a virtual machine.
### (1) Install Conda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
``` 
```bash
bash Anaconda3-2021.11-Linux-x86_64.sh
```
### (2) Create & Access Virtual Environment
Create enviornment named ***adp*** that has ***python 3.6.13*** installed
```bash
conda create -n adp python=3.6.13
```
```bash
conda activate adp
```
### (3) Download Java VM
Download Java Virtual Machine
```bash
sudo apt-get update
```
```bash
sudo apt-get install default-jdk
```
```bash
java -version
```
### (4) Download CPLEX
Visit [CPLEX website](https://www.ibm.com/ca-en/products/ilog-cplex-optimization-studio) and login using an academic/professional account. Download IBM ILOG CPLEX Optimization Studio bin file called ***cplex_studio1210.linux-x86-64.bin***.
Then you may upload the bin file to your VM and run the following commands:
```bash
chmod +x cplex_studio1210.linux-x86-64.bin
```
```bash
mkdir CPLEX
```
```bash
./cplex_studio1210.linux-x86-64.bin
```
When asked during installation where to setup CPLEX, you may provide the path to your CPLEX folder:
```bash
/PATH/TO/CPLEX
```
Finally, you may setup CPLEX via the following:
```bash
python CPLEX/python/setup.py install --user
```
### (5) Install Necessary Packages
You may run the `requirements.txt` file provided via:
```bash
pip install -r requirements.txt
```
Conversely, you may manually install the necessary packages via:
```bash
pip install Keras==2.2.4
pip install pandas==1.1.5
pip install numpy==1.19.5
pip install matplotlib==3.3.4
pip install tensorflow==1.15.0
pip install tqdm
pip install networkx
```

## Citing this Work
To cite this work, please use the following:
```bash
@article{dehghan2023dynamic,
  title={Dynamic AGV Task Allocation in Intelligent Warehouses},
  author={Dehghan, Arash and Cevik, Mucahit and Bodur, Merve},
  journal={arXiv preprint arXiv:2312.16026},
  year={2023}
}
```
