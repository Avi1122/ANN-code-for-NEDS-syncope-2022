<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email
-->





<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Novel Machine Learning Algorithms to Risk Stratify Patients with Syncope Presenting to the Emergency Department </h3>
<br />
  <p align="center">
    <br />
     <a href="https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022">
    <img src="images/Watershed.jpg" align="right"  width="500" height="400"/>
     </a>
    <a href="https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022"><strong>Explore the docs » </strong></a>
    <br />
    <br />
    <a href="https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022/issues">Report Bug</a>
    ·
    <a href="https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

[![Novel Machine Learning Algorithms to Risk Stratify Patients with Syncope Presenting to the Emergency Department ][product-screenshot]](https://example.com)

The aim of this research is to Predict Length of Stay for Patients Admitted with Syncope from the Emergency Department: 
    <ol>
<li>NEDS data(2016-2019) is filtered to exclude the population sample of Age (less than) 18 and those without mortality data. The NEDS data contains the ICD-10-CM diagnosis codes. However, for our aim, these codes are used to compute 31 Elixhauser comorbidity indices (ECI) to utilize them in the ML algorithm as a representation of personalized cardiovascular risk factors. ECI is a validated method for categorizing patient-specific comorbidities in a large administrative database based on ICD diagnosis codes. Also, we categorized LoS into short stay (negative class) and long stay (positive class). The modified input and output variables are</li>
<li>Input Variables (X) – Patient demographics, Month and week of admission, Rurality based on metropolitan statistical area (MSA) status, whether the ED is affiliated with an academic institution or community hospital, and 31 Elixhauser comorbidity indices.</li>
<li>Outcomes (Y) – short stay vs Long stay</li>
</ol>

As illustrated in the above picture, We use Multi layered Artificial Neural networks for predicting Length of Stay.

### Built With
The major frameworks that we used to built our project are:
* [Python](https://www.python.org/downloads/release/python-380/)
* [Tensorflow](https://www.tensorflow.org/api_docs)
* [Keras](https://keras.io)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites
<p>
Our work environment includes: <br />
cuda 11.7 <br />
gcc  9.4.0 <br />
cudnn_version 8 <br />
Nvidia driver 515.65.07 <br />
4 x NVIDIA A100-SXM 80 GB GPU machines <br />
anaconda3/2021.05 <br />

However, feel free to try different combinations using this [this](https://www.tensorflow.org/install/source#tested_build_configurations) guide which hints the tensorflow conpatibility. Assuming that you have a working combination of the prerequisites, 
<br />
</p>
1.We suggest you to start with a new conda environment. Say, 

```sh
conda create --name los_prediction python=3.8
```
2.Activate the environment using

```sh
conda activate los_prediction
```

### Installation

1. Clone the repo
```sh
git clone https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022.git
```
2. Install dependency packages using the *requirements.txt* file attached in this repo using
```sh
pip install -r requirements.txt
```

<!-- USAGE -->
## Usage

All the supported data is in the Git along with the scripts. TheScripts are executed in this order:
<ul>
<li><em>Emergency_data_preprossing.py</em> Used to create the dataset used for modeling. However, the data is not upsampled here </li>
<li><em>Emergency_data_upsampling.py</em> The output of Emergency_data_preprossing.py is the input to this file. This program upsamples the data for different LOS thresholds used for analysis</li>
<li><em>Emergency_model.py</em> This file contains model as well as the analysis. This file takes an additional argument as input which is LOS threshold of classification. Say threshold 1 implies that you classify LOS as (less than 1 day) vs (more than 1 day).</li>
</ul>

Make sure that the shell scripts in the repo have necessary permisions to execute.
```sh
chmod +x *.*
```
You can execute the files using this syntax.<br />
```sh
python Emergency_data_preprossing.py
```
or <br />
```sh
python Emergency_data_upsampling.py
```
or <br />
```sh
python Emergency_model.py 2
```


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Mudireddy Avinash Reddy - [@linkedin](https://www.linkedin.com/in/mudireddy-avinash-reddy-367121a2/) 

Project Link: [https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022](https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [IIAI](https://www.iiai.uiowa.edu/)
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=flat-square
[contributors-url]: https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=flat-square
[forks-url]: https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=flat-square
[stars-url]: https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=flat-square
[issues-url]: https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=flat-square
[license-url]: https://github.com/kagocchi28/ANN-code-for-NEDS-syncope-2022/blob/master/LICENSE.txt
[product-screenshot]: images/diag.png
